#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
from scene.gaussian_pbr_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.image_utils import linear_to_srgb
from einops import rearrange

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
    
    xyz = pc.get_xyz # (N, 3) 
    view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3)
    diffuse   = pc.get_diffuse # (N, 3)
    specular  = pc.get_specular # (N, 3) 
    roughness = pc.get_roughness # (N, 1)
    
    # render xyz buffer
    render_xyz, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = xyz,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space

    render_normal = allmap[2:5]
    from icecream import ic

    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
   
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets = {
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    }
    
    # render G buffer
    in_dict = {
        "diffuse": diffuse, 
        "specular": specular, 
        "roughness": roughness.repeat(1, 3)
    }
    out_buffer_dict = {k: rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = in_dict[k],
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)[0] for k in in_dict.keys() if in_dict[k] is not None}
    out_buffer_dict['xyz'] = render_xyz
    out_buffer_dict['normal'] = render_normal

    # in_dict['specular'].retain_grad()
    # ic(in_dict['specular'])
    # ic(means3D)
    # ic(opacity)
    # ic(scales)
    # ic(rotations)
    # ic(out_buffer_dict['specular'])
    # out_buffer_dict['specular'].sum().backward()
    # ic(in_dict['specular'].grad)

    # use pixel_alpha to accelerate at test time
    # import pdb; pdb.set_trace()
    pixel_alpha = rearrange(render_alpha, 'C H W -> (H W) C')
    mask = pixel_alpha[..., 0] > 0.5
    if mask.sum() == 0:
        mask = torch.ones_like(pixel_alpha[..., 0]).bool().cuda()
    pixel_xyz = rearrange(out_buffer_dict['xyz'], 'C H W -> (H W) C')[mask, :]
    pixel_normal = rearrange(out_buffer_dict['normal'], 'C H W -> (H W) C')[mask, :]
    pixel_diffuse = rearrange(out_buffer_dict['diffuse'], 'C H W -> (H W) C')[mask, :]
    pixel_specular = rearrange(out_buffer_dict['specular'], 'C H W -> (H W) C')[mask, :]
    pixel_roughness = rearrange(out_buffer_dict['roughness'], 'C H W -> (H W) C')[mask, :]
    pixel_view_pos = viewpoint_camera.camera_center.repeat(pixel_xyz.shape[0], 1) # (N, 3) 
    
    param1=pixel_xyz[None, None, :, :]
    param2=pixel_normal[None, None, :, :]
    param3=pixel_diffuse[None, None, :, :]
    param4=pixel_specular[None, None, :, :]
    param5=pixel_roughness[None, None, :, 0:1]
    pixel_color, brdf_pkg = pc.brdf_mlp.shade(param1, param2, param3, param4, param5, pixel_view_pos[None, None, :, :])

    new_pixel_color = torch.ones((pixel_alpha.shape[0], pixel_color.shape[-1])).cuda() if bg_color[0] == 1.0 else torch.zeros((pixel_alpha.shape[0], pixel_color.shape[-1])).cuda()
    new_pixel_color[mask, :] = pixel_color.squeeze()
    rendered_image = rearrange(new_pixel_color, '(H W) C -> C H W', H=out_buffer_dict['xyz'].shape[1], W=out_buffer_dict['xyz'].shape[2])
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
    rendered_image = linear_to_srgb(rendered_image)

    # param1.retain_grad()
    # param2.retain_grad()
    # param3.retain_grad()
    # param4.retain_grad()
    # pixel_color.retain_grad()
    # rendered_image.sum().backward()
    # ic(param1)
    # ic(param1.grad)
    # ic(param2.grad)
    # ic(param3.grad)
    # ic(pixel_color.grad)

    rets.update({"render": rendered_image,
            "diffuse": linear_to_srgb(out_buffer_dict['diffuse']),
            "specular": linear_to_srgb(out_buffer_dict['specular']),
            "roughness": linear_to_srgb(out_buffer_dict['roughness']),
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            })

    return rets