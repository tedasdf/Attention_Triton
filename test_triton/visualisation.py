import torch
import matplotlib.pyplot as plt


def make_index_tensor(batch, num_heads, seq_len, dim, device="cpu"):
    q = torch.empty((batch, num_heads, seq_len, dim), dtype=torch.int64, device=device)

    for b in range(batch):
        for h in range(num_heads):
            for s in range(seq_len):
                for d in range(dim):
                    # Encodes coordinates into one readable integer
                    q[b, h, s, d] = b * 10_000_000 + h * 100_000 + s * 1_000 + d

    return q


def decode_value(val):
    b = val // 10_000_000
    rem = val % 10_000_000
    h = rem // 100_000
    rem = rem % 100_000
    s = rem // 1_000
    d = rem % 1_000
    return int(b), int(h), int(s), int(d)


def print_tile_info(tile):
    rows, cols = tile.shape
    print("\nDecoded coordinates:")
    for i in range(rows):
        row_items = []
        for j in range(cols):
            val = int(tile[i, j].item())
            row_items.append(str(decode_value(val)))
        print(" | ".join(row_items))


def visualize_flat_tensor(q):
    q_np = q.cpu().numpy()
    flat = q_np.reshape(-1)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.imshow(flat[None, :], aspect="auto")
    ax.set_title("Flat memory layout")
    ax.set_xlabel("Flat offset")
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def get_q_tile(
    q,
    b,
    h,
    pid,
    BLOCK_SIZE_M,
    BLOCK_DMODEL,
):
    stride_qb, stride_qh, stride_qs, stride_qd = q.stride()

    # Global sequence indices for this tile
    seq_idx = pid * BLOCK_SIZE_M + torch.arange(BLOCK_SIZE_M, device=q.device)
    d_idx = torch.arange(BLOCK_DMODEL, device=q.device)

    # Flat offsets, same spirit as Triton pointer arithmetic
    base = b * stride_qb + h * stride_qh
    offsets = base + seq_idx[:, None] * stride_qs + d_idx[None, :] * stride_qd

    flat_q = q.reshape(-1)
    tile = flat_q[offsets]

    return tile, offsets, seq_idx, d_idx


if __name__ == "__main__":
    device = "cpu"

    batch = 1
    num_heads = 4
    seq_len = 8
    dim = 4

    BLOCK_SIZE_M = 4
    BLOCK_DMODEL = 4

    q = make_index_tensor(batch, num_heads, seq_len, dim, device=device)

    print("Q shape:", q.shape)
    print("Q strides:", q.stride())

    visualize_flat_tensor(q)

    b = 0
    h = 2
    pid = 1

    tile, offsets, seq_idx, d_idx = get_q_tile(
        q=q,
        b=b,
        h=h,
        pid=pid,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )

    print("\nSelected tile values:")
    print(tile)

    print("\nFlat offsets:")
    print(offsets)

    print("\nSequence indices used:")
    print(seq_idx)

    print("\nD indices used:")
    print(d_idx)

    print_tile_info(tile)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(q[0, :, :, 0].cpu().numpy(), aspect="auto")
    ax1.set_title("Q[:, :, :, 0] view")
    ax1.set_xlabel("seq")
    ax1.set_ylabel("head")

    ax2.imshow(tile.cpu().numpy(), aspect="auto")
    ax2.set_title(f"Loaded Q tile (b={b}, h={h}, pid={pid})")
    ax2.set_xlabel("d")
    ax2.set_ylabel("tile row")

    plt.tight_layout()
    plt.show()
