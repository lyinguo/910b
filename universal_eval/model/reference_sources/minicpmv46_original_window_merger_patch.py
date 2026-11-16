"""Exact saved TransformersVLM per-image window-merger compatibility patch."""
import torch as _torch
def _merger_forward_per_image(self, hidden_states, target_sizes, **kwargs):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    _dev = hidden_states.device
    _h, _w = self.window_kernel_size
    _widx, _cuseq, _wmax = self.get_window_index(target_sizes, kwargs=kwargs)
    _widx = _widx.to(_dev)
    hidden_states = hidden_states[:, _widx, :]
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        cu_seqlens=_cuseq.to(_dev),
        max_seqlen=_wmax,
    )
    hidden_states = hidden_states[:, _torch.argsort(_widx), :]
    hidden_states = residual + hidden_states

    _emb = hidden_states.shape[-1]
    _grids = target_sizes
    _prod = (_grids[:, 0] * _grids[:, 1]).tolist()
    _starts = [0]
    for _p in _prod:
        _starts.append(_starts[-1] + _p)
    _outs = []
    for _i in range(_grids.shape[0]):
        _sl = slice(_starts[_i], _starts[_i + 1])
        _img = hidden_states[:, _sl, :]
        _ih, _iw = int(_grids[_i, 0].item()), int(_grids[_i, 1].item())
        _mh, _mw = _ih // _h, _iw // _w
        _pk = _img.view(1, _mh, _h, _mw, _w, _emb).permute(0, 1, 3, 2, 4, 5)
        _flat = _pk.reshape(_mh * _mw, _h * _w * _emb)
        _res = _pk.reshape(_mh * _mw, _h * _w, _emb).mean(dim=1)
        _x = self.pre_norm(_flat)
        _x = self.linear_1(_x)
        _x = self.act(_x)
        _x = self.linear_2(_x)
        _outs.append(_x + _res)
    return _torch.cat(_outs, dim=0).unsqueeze(0)
