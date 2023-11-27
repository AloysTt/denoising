#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import subprocess as sp
import json
import numpy as np
from PIL import Image
from argparse import ArgumentParser

    
def config_targets_scene(args):
    """Modifies Tungsten scene file to save normal render for targets."""
    
    # Load JSON scene file
    if not args.scene_path.endswith('.json'):
        raise ValueError('Scene file must be in JSON format')
    with open(args.scene_path, 'r') as fp:
        scene = json.load(fp)

    # Save in low dynamic range
    scene['camera']['tonemap'] = 'reinhard'
    scene['renderer']['output_file'] = 'target.png'
        
    # Update resolution, if requested
    if args.resolution:
        res = scene['camera']['resolution']
        if isinstance(res, int):
            w, h = res, res
        else:
            w, h = res[0], res[1]
        ratio_preserved = w / h == args.resolution[0] / args.resolution[1]
        assert ratio_preserved, 'Resizing image with ratio that doesn\'t match reference'
        scene['camera']['resolution'] = list(args.resolution)

    # Update SPP count
    scene['renderer']['spp'] = args.spp

    # Save target scene configuration
    scene_dir = os.path.dirname(os.path.splitext(args.scene_path)[0])
    target_file = f'scene_target.json'
    target_path = os.path.join(scene_dir, target_file)
    with open(target_path, 'w') as fp:
        json.dump(scene, fp, indent=2)

    return target_path
    

def batch_render(target_path, args):
    """Renders scene N times and save to output directory."""

    # Create render directory, if nonexistent
    subdirs = ['target']
    subdirs_paths = [os.path.join(args.output_dir, s) for s in subdirs]
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        [os.mkdir(s) for s in subdirs_paths]
    
    # File extensions
    file_ext = '.png'

    # Batch render
    for i in range(args.nb_renders):
        img_id = '{0:04d}'.format(i + 1)
        
        # Render with Tungsten
        seeds = np.random.randint(0, 1e6, size=2)
        render_target_cmd = f'{args.tungsten} -s {seeds[1]} -d {args.output_dir} {target_path}'
        sp.call(render_target_cmd.split())

        # Create renaming/moving commands
        mv_imgs = []
        for name, subdir_path in zip(subdirs, subdirs_paths):
            filename = os.path.join(args.output_dir, name + file_ext)
            dest = os.path.join(subdir_path, f'{img_id}_{name}{file_ext}')
            mv_imgs.append(f'mv {filename} {dest}')
            
        # Call
        for mv in mv_imgs:
            sp.call(mv.split())

    # Move reference images
    scene_root = os.path.dirname(target_path)
    if args.hdr_buffers or args.hdr_targets:
        if args.resolution:
            print('Warning: Could not resize reference image, do it manually')
        mv_ref_hdr = f'cp {scene_root}/TungstenRender.exr {args.output_dir}/reference.exr'
        sp.call(mv_ref_hdr.split())
    else:
        if args.resolution:
            ref_ldr = Image.open(f'{scene_root}/TungstenRender.png')
            ref_ldr = ref_ldr.resize(tuple(args.resolution), Image.BILINEAR)
            ref_ldr.save(f'{scene_root}/TungstenRender.png')
        
        mv_ref_ldr = f'cp {scene_root}/TungstenRender.png {args.output_dir}/reference.png'
        sp.call(mv_ref_ldr.split())


def parse_args():
    """Command-line argument parser for generating scenes."""

    # New parser
    parser = ArgumentParser(description='Monte Carlo rendering generator')

    # Rendering parameters
    parser.add_argument('-t', '--tungsten', help='tungsten renderer full path', default='tungsten', type=str)
    parser.add_argument('-d', '--scene-path', help='scene root path', type=str)
    parser.add_argument('-r', '--resolution', help='image resolution (w, h)', nargs='+', type=int)
    parser.add_argument('-s', '--spp', help='sample per pixel', default=16, type=int)
    parser.add_argument('-n', '--nb-renders', help='number of renders', default=10, type=int)
    parser.add_argument('--hdr-buffers', help='save buffers as hdr images', action='store_true')
    parser.add_argument('--hdr-targets', help='save targets as hdr images', action='store_true')
    parser.add_argument('-o', '--output-dir', help='output directory', default='../../data/renders', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    """Creates scene files."""

    # Parse render parameters and create scene file
    args = parse_args()
    target_path = config_targets_scene(args)
    batch_render(target_path, args)
