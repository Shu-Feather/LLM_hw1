# -*- coding: utf-8 -*-
class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.id2bytes = {}
        for idx in range(256):
            self.id2bytes[idx] = bytes([idx])

        self.bytes2id = {v: k for k, v in self.id2bytes.items()}  

    def _get_stats(self, ids):
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            counts[pair] = counts.get(pair, 0) + 1
        
        return counts

    def _merge_pair(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
            
        return new_ids

    def train(self, text, vocab_size):
        data = list(text.encode("utf-8"))
        ids = data.copy() 

        next_id = 256 
        
        while len(self.id2bytes) < vocab_size:
            stats = self._get_stats(ids)
            if not stats:
                break

            best_pair = max(stats, key=lambda p: stats[p])
            
            new_id = next_id
            self.merges[best_pair] = new_id
            merged_bytes = self.id2bytes[best_pair[0]] + self.id2bytes[best_pair[1]]
            self.id2bytes[new_id] = merged_bytes
            self.bytes2id[merged_bytes] = new_id
            
            ids = self._merge_pair(ids, best_pair, new_id)
            next_id += 1

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        for pair, idx in sorted(self.merges.items(), key=lambda x: -len(self.id2bytes[x[0][0]])):
            ids = self._merge_pair(ids, pair, idx)

        return ids

    def decode(self, ids):
        byte_stream = b"".join(self.id2bytes[idx] for idx in ids)

        return byte_stream.decode("utf-8", errors="replace")


if __name__ == "__main__":
    sample = "low lower lowest low"
    tok = Tokenizer()
    tok.train(sample, vocab_size=20)

    seq = tok.encode("low lowest")
    print("Encoded IDs:", seq)

    txt = tok.decode(seq)
    print("Decoded text:", txt)
