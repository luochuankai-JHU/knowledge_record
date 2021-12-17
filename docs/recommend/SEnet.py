    def _se_block(self, id_concat_emb, cont_concat_emb, lr_x):
        """
        SE-net block
        """
        num_id_fea = id_concat_emb.shape[1] / self._emb_dim
        id_emb_col = fluid.layers.reshape(id_concat_emb, shape=[-1, num_id_fea, self._emb_dim])
        id_emb_sqz = fluid.layers.reduce_mean(id_emb_col, dim=-1)
        concat_emb_w0 = self._concat_emb([id_emb_sqz, cont_concat_emb])
        fields_num = concat_emb_w0.shape[1]
        reducer_dim = int(fields_num / self._senet_ratio)
        scale = self._init_range / (reducer_dim ** 0.5)
        fc1 = self._fc_layer(concat_emb_w0, reducer_dim, 'relu', scale)
        scale = self._init_range / (fields_num ** 0.5)
        fc2 = self._fc_layer(fc1, fields_num, 'sigmoid', scale)
        w_id = fluid.layers.slice(fc2, axes=[0, 1], starts=[0, 0], ends=[self._int_max, num_id_fea])
        w_cont = fluid.layers.slice(fc2, axes=[0, 1], starts=[0, num_id_fea], \
            ends=[self._int_max, self._int_max])
        cont_mul = cont_concat_emb * w_cont
        id_mul = id_emb_col * w_id
        id_mul_out = fluid.layers.reshape(id_mul, shape=[-1, num_id_fea * self._emb_dim])
        return id_mul_out, cont_mul

    def _senet(self, id_concat_emb, cont_concat_emb, lr_x, skip_connect=False):
        """
        _senet
        """
        id_mul_out1, cont_mul1 = self._se_block(id_concat_emb, cont_concat_emb, lr_x)
        id_mul_out2, cont_mul2 = self._se_block(id_mul_out1, cont_mul1, lr_x)
        id_mul_out3, cont_mul3 = self._se_block(id_mul_out2, cont_mul2, lr_x)
        if skip_connect == True:
            id_mul_out3 += id_concat_emb
            cont_mul3 += cont_concat_emb
        out = self._concat_emb([cont_mul3, id_mul_out3])
        nn_nodes = min(400, out.shape[1]/2)
        scale = self._init_range / (nn_nodes ** 0.5)
        se_out = self._fc_layer(out, nn_nodes, 'relu', scale)
        return se_out
