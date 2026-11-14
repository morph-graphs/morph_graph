import torch


def get_qwen3_vl_rope_index(
    input_ids,
    video_grid_thw,
    attention_mask,
    spatial_merge_size,
    video_token_id,
    vision_start_token_id,
):
    """Build Qwen3-VL M-RoPE position ids from token ids and video grids."""
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)

    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    mrope_position_deltas = []
    video_index = 0
    attention_mask = attention_mask.to(total_input_ids.device)

    for i, input_ids in enumerate(total_input_ids):
        input_ids = input_ids[attention_mask[i] == 1]
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
        vision_tokens = input_ids[vision_start_indices + 1]
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list = []
        st = 0
        remain_videos = video_nums

        for _ in range(video_nums):
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1

            t, h, w = (
                video_grid_thw[video_index][0],
                video_grid_thw[video_index][1],
                video_grid_thw[video_index][2],
            )
            video_index += 1
            remain_videos -= 1
            ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # t_index is always 0 because llm_grid_t is always 1; timestamps
            # carry temporal information for videos in this code path.
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def get_qwen3_vl_retrieved_position_ids(
    init_len,
    block_indices,
    adaptive_block_sizes,
    visual_block_size,
    grid_hw,
    device,
    keep_index=None,
):
    """Build Qwen3-VL position ids for packed retrieved-frame KV from lengths only.

    keep_index: optional LongTensor of patch indices retained by compressed retrieval.
    Kept patches keep their ORIGINAL (h, w) grid coordinates -- the point of striding is
    to drop tokens, not to renumber the survivors into a denser grid, which would tell the
    model the frame was smaller than it was.
    """
    grid_h, grid_w = grid_hw
    assert visual_block_size == grid_h * grid_w, f"{visual_block_size} vs. {grid_h} * {grid_w}"

    position_chunks = []
    current_pos = init_len
    t_index = torch.zeros(grid_h * grid_w, dtype=torch.long, device=device)
    h_index = torch.arange(grid_h, dtype=torch.long, device=device).view(-1, 1).expand(-1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long, device=device).view(1, -1).expand(grid_h, -1).flatten()
    visual_grid_pos = torch.stack([t_index, h_index, w_index]).view(3, 1, -1)
    kept_grid_pos = visual_grid_pos
    if keep_index is not None:
        kept_grid_pos = visual_grid_pos[:, :, keep_index.to(device)]
    # Offset of <vision_end> within a frame, measured on the FULL grid. Using the kept
    # maximum instead would shrink each frame's positional footprint whenever the stride
    # drops the last row or column, shifting every later frame.
    grid_span = int(visual_grid_pos.max())

    for block_idx in block_indices:
        timestamp_len = adaptive_block_sizes[block_idx] - visual_block_size - 2
        assert timestamp_len >= 0

        text_before_visual_len = timestamp_len + 1  # timestamp text + <vision_start>
        text_pos = torch.arange(
            current_pos,
            current_pos + text_before_visual_len,
            dtype=torch.long,
            device=device,
        ).view(1, 1, -1)
        position_chunks.append(text_pos.expand(3, 1, -1))

        visual_pos = kept_grid_pos + current_pos + text_before_visual_len
        position_chunks.append(visual_pos)

        end_pos = grid_span + current_pos + text_before_visual_len + 1
        position_chunks.append(torch.full((3, 1, 1), end_pos, dtype=torch.long, device=device))
        current_pos = end_pos + 1

    return torch.cat(position_chunks, dim=-1)
