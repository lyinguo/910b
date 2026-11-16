from transformers import PretrainedConfig


class FalconOCRConfig(PretrainedConfig):
    model_type = "falcon_ocr"

    def __init__(
        self,
        dim: int = 768,
        n_layers: int = 22,
        n_heads: int = 16,
        head_dim: int = 64,
        n_kv_heads: int = 8,
        vocab_size: int = 65536,
        ffn_dim: int = 2304,
        norm_eps: float = 1e-5,
        max_seq_len: int = 8192,
        rope_theta: int = 10000,
        channel_size: int = 3,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 1,
        img_id: int = 227,
        eos_id: int = 11,
        image_cls_token_id: int = 244,
        image_mask_token_id: int = 243,
        image_reg_1_token_id: int = 245,
        image_reg_2_token_id: int = 246,
        image_reg_3_token_id: int = 247,
        image_reg_4_token_id: int = 248,
        img_start_id: int = 229,
        img_end_id: int = 230,
        img_row_sep_id: int = 228,
        vid_start_id: int = 231,
        vid_end_id: int = 232,
        frame_sep_id: int = 233,
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.ffn_dim = ffn_dim
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.channel_size = channel_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.img_id = img_id
        self.eos_id = eos_id
        self.image_cls_token_id = image_cls_token_id
        self.image_mask_token_id = image_mask_token_id
        self.image_reg_1_token_id = image_reg_1_token_id
        self.image_reg_2_token_id = image_reg_2_token_id
        self.image_reg_3_token_id = image_reg_3_token_id
        self.image_reg_4_token_id = image_reg_4_token_id
        self.img_start_id = img_start_id
        self.img_end_id = img_end_id
        self.img_row_sep_id = img_row_sep_id
        self.vid_start_id = vid_start_id
        self.vid_end_id = vid_end_id
        self.frame_sep_id = frame_sep_id
        super().__init__(**kwargs)
