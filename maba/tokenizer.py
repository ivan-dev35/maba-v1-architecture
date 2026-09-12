from typing import List, Dict, Optional

class Tokenizer:
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self.id_to_tok: Dict[int, str] = {}
        self.tok_to_id: Dict[str, int] = {}
        self._build_vocab()

    def _build_vocab(self):
        specials = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for i, t in enumerate(specials):
            self.id_to_tok[i] = t
            self.tok_to_id[t] = i

        for b in range(256):
            t = f"<0x{b:02X}>"
            i = 4 + b
            self.id_to_tok[i] = t
            self.tok_to_id[t] = i

        words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "def", "class", "return", "import", "from", "for", "while", "if", "else", "elif",
            "int", "float", "double", "void", "bool", "auto", "const", "struct", "template",
            "include", "namespace", "std", "cout", "vector", "string", "tensor", "matrix",
            "model", "forward", "backward", "loss", "optim", "grad", "layer", "hidden",
            "в", "и", "не", "на", "я", "быть", "он", "с", "что", "а", "по", "это",
            "она", "этот", "к", "но", "они", "мы", "как", "из", "у", "который", "то",
            "за", "свой", "весь", "год", "от", "так", "о", "для", "ты", "же", "все",
            "модель", "архитектура", "параметр", "слой", "внимание", "память", "тензор"
        ]

        cid = 260
        for w in words:
            if cid >= self.vocab_size:
                break
            if w not in self.tok_to_id:
                self.id_to_tok[cid] = w
                self.tok_to_id[w] = cid
                cid += 1

        while cid < self.vocab_size:
            t = f"<sub_{cid}>"
            self.id_to_tok[cid] = t
            self.tok_to_id[t] = cid
            cid += 1

        self.max_word_len = max((len(k) for k, v in self.tok_to_id.items() if v >= 260), default=0)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        toks = [self.BOS_ID] if add_bos else []
        i = 0
        n = len(text)
        max_l = getattr(self, "max_word_len", 32)
        while i < n:
            matched = False
            for length in range(min(max_l, n - i), 0, -1):
                sub = text[i:i + length]
                tid = self.tok_to_id.get(sub)
                if tid is not None and tid >= 260:
                    toks.append(tid)
                    i += length
                    matched = True
                    break
            if not matched:
                b = text[i].encode("utf-8")
                for byte in b:
                    toks.append(4 + byte)
                i += 1
        if add_eos:
            toks.append(self.EOS_ID)
        return toks

    def decode(self, ids: List[int]) -> str:
        b = bytearray()
        for i in ids:
            if i in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID):
                continue
            if 4 <= i <= 259:
                b.append(i - 4)
            else:
                tok = self.id_to_tok.get(i, f"<{i}>")
                b.extend(tok.encode("utf-8", errors="ignore"))
        return b.decode("utf-8", errors="replace")

MabaTokenizer = Tokenizer
