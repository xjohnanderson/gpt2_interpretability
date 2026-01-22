class ResidualHookManager:
    def __init__(self, model):
        self.model = model
        self.attn_outputs = []
        self.mlp_outputs = []
        self.handles = []

    def _save_attn(self, module, input, output):
        self.attn_outputs.append(output[0])

    def _save_mlp(self, module, input, output):
        self.mlp_outputs.append(output)

    def register_hooks(self):
        for block in self.model.h:
            self.handles.append(block.attn.register_forward_hook(self._save_attn))
            self.handles.append(block.mlp.register_forward_hook(self._save_mlp))

    def clear(self):
        self.attn_outputs.clear()
        self.mlp_outputs.clear()

    def remove_hooks(self):
        for h in self.handles:
            h.remove()