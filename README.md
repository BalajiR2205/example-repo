def greedy_decode(self, src_ids, max_len=64):
        self.eval()
        src_key_padding_mask = (src_ids == PAD)
        memory = self.transformer.encoder(
            self.pos_enc(self.src_emb(src_ids)),
            src_key_padding_mask=src_key_padding_mask
        )

        B = src_ids.size(0)
        ys = torch.full((B, 1), SOS, dtype=torch.long, device=src_ids.device)

        for _ in range(max_len):
            tgt_mask = torch.triu(torch.ones(ys.size(1), ys.size(1), device=ys.device) * float("-inf"), diagonal=1)
            tgt = self.pos_enc(self.tgt_emb(ys))
            out = self.transformer.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(ys == PAD),
                memory_key_padding_mask=src_key_padding_mask
            )
            logits = self.out(out[:, -1, :])  # [B, V]
            next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == EOS).all():
                break
        return ys  # includes SOS ... EOS