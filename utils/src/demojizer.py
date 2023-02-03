import emoji


def legacy_demojizer(x: str) -> str:
    return "".join(filter(lambda ch: not emoji.is_emoji(ch), x))


class Demojizer:
    """
    based on:
    https://github.com/carpedm20/emoji/blob/d8bbfe455c6fcd12b96ed1dce6e0978fe7a47431/emoji/core.py#L141
    """

    def _get_search_tree(self):
        _SEARCH_TREE = {}
        for emj in emoji.unicode_codes.EMOJI_DATA:
            sub_tree = _SEARCH_TREE
            lastidx = len(emj) - 1
            for i, char in enumerate(emj):
                if char not in sub_tree:
                    sub_tree[char] = {}
                sub_tree = sub_tree[char]
                if i == lastidx:
                    sub_tree["data"] = emoji.unicode_codes.EMOJI_DATA[emj]
        return _SEARCH_TREE

    def __init__(self) -> None:
        self.search_tree = self._get_search_tree()

    def __call__(self, string: str, replace_str: str):
        result = []
        i = 0
        length = len(string)
        state = 0
        while i < length:
            consumed = False
            char = string[i]
            if char in self.search_tree:
                j = i + 1
                sub_tree = self.search_tree[char]
                while j < length and string[j] in sub_tree:
                    sub_tree = sub_tree[string[j]]
                    j += 1
                if "data" in sub_tree:
                    state = 1
                    consumed = True
                    result.append(replace_str)
                    i = j - 1
                else:
                    state = 0
            elif state == 1:
                if char.isspace():
                    consumed = True
                else:
                    state = 0

            if not consumed and char != "\ufe0e" and char != "\ufe0f":
                result.append(char)
            i += 1

        return "".join(result)
