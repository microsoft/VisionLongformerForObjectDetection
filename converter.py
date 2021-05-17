# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
"""
Helper script to convert classification models from vision longformer to be used with the Detectron2 version.
https://github.com/microsoft/vision-longformer
"""
import math
import torch

from train_net import setup, Trainer
from detectron2.engine import default_argument_parser, default_setup


def resize_pos_embed_1d(posemb, shape_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_old = posemb.shape[1]
    if ntok_old>1:
        ntok_new = shape_new[1]
        posemb_grid = posemb.permute(0, 2, 1).unsqueeze(dim=-1)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=[ntok_new, 1], mode='bilinear')
        posemb_grid = posemb_grid.squeeze(dim=-1).permute(0, 2, 1)
        posemb = posemb_grid
    return posemb


def resize_pos_embed_2d(posemb, shape_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = shape_new[0]
    gs_old = int(math.sqrt(len(posemb)))  # 2 * w - 1
    gs_new = int(math.sqrt(ntok_new))  # 2 * w - 1
    posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(gs_new * gs_new, -1)
    return posemb_grid


def main():
    args = default_argument_parser()
    args.add_argument("--source_model", default="", type=str, help="Path or url to the model to convert")
    args.add_argument("--output_model", default="", type=str, help="Path where to save the converted model")
    args = args.parse_args()
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    target_state_dict = model.state_dict()

    if "vit" in cfg.MODEL.BACKBONE.NAME:
        checkpoint = torch.load(args.source_model, map_location="cpu")
        model_to_convert = checkpoint["net"]
        model_converted = {}
        for key in model_to_convert.keys():
            val = model_to_convert[key].detach()
            new_key = "backbone.bottom_up." + key.replace('module.', '')

            if new_key in target_state_dict:
                if 'cls_token' in key:
                    shape_new = target_state_dict[new_key].shape
                    Nglo = shape_new[1]
                    model_converted[new_key] = val.expand(-1, Nglo, -1)
                    print("EXPAND", key, "->", new_key, " Nglo to {}".format(Nglo))
                elif 'attn.proj' in key:
                    model_converted[new_key] = val
                    print(key, "->", new_key)
                    proj_global_name = new_key.replace('attn.proj', 'attn.proj_global')
                    if proj_global_name in target_state_dict:
                        model_converted[proj_global_name] = val
                        print(key, "->", proj_global_name)
                elif 'x_pos_embed' in key or 'y_pos_embed' in key:
                    shape_old = val.shape
                    shape_new = target_state_dict[new_key].shape
                    if shape_old != shape_new:
                        new_val = resize_pos_embed_1d(val, shape_new)
                    else:
                        new_val = val
                    if shape_new == new_val.shape:
                        model_converted[new_key] = new_val
                        print("[RESIZE] {} {} -> {} {}".format(key, shape_old, new_key, shape_new))
                    else:
                        print("[WARNING]","{} {} != {} {}, skip".format(key, new_val.shape, new_key, shape_new))
                elif 'local_relative_position_bias_table' in key:
                    shape_old = val.shape
                    shape_new = target_state_dict[new_key].shape
                    if shape_old != shape_new:
                        new_val = resize_pos_embed_2d(val, shape_new)
                    else:
                        new_val = val
                    if shape_new == new_val.shape:
                        model_converted[new_key] = new_val
                        print("[RESIZE] {} {} -> {} {}".format(key, shape_old, new_key, shape_new))
                    else:
                        print("[WARNING]", "{} {} != {} {}, skip".format(key, new_val.shape, new_key, shape_new))
                elif 'relative_position_index' in key:
                    shape_old = val.shape
                    shape_new = target_state_dict[new_key].shape
                    print("Skipping converting {} {} to {} {}".format(key, shape_old, new_key, shape_new))
                else:
                    print(key, "->", new_key)
                    model_converted[new_key] = val
            elif 'attn.qkv' in key:
                assert len(val) // 3, "First dimension of {} is not divided by 3!"
                dim = len(val) // 3
                query_name = new_key.replace('attn.qkv', 'attn.query')
                if query_name in target_state_dict:
                    model_converted[query_name] = val[:dim]
                    print(key, "->", query_name)
                query_global_name = new_key.replace('attn.qkv', 'attn.query_global')
                if query_global_name in target_state_dict:
                    model_converted[query_global_name] = val[:dim]
                    print(key, "->", query_global_name)
                kv_name = new_key.replace('attn.qkv', 'attn.kv')
                if kv_name in target_state_dict:
                    model_converted[kv_name] = val[dim:]
                    print(key, "->", kv_name)
                kv_global_name = new_key.replace('attn.qkv', 'attn.kv_global')
                if kv_global_name in target_state_dict:
                    model_converted[kv_global_name] = val[dim:]
                    print(key, "->", kv_global_name)

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
