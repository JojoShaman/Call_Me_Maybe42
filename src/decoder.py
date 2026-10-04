from pydantic import BaseModel, ConfigDict, PrivateAttr
from llm_sdk import Small_LLM_Model
from typing import Any, Callable
import json
from .errors import CallMeError
import numpy as np


class Decoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model
    _id_to_token: dict[int, str] = (
        PrivateAttr(default_factory=dict[int, str]))

    def model_post_init(self, context: Any) -> None:
        path = self.model.get_path_to_vocab_file()
        with open(path, encoding='utf-8') as f:
            vocab: dict[str, int] = json.load(f)
        self._id_to_token = {
            id_token: self.model.decode([id_token])
            for id_token in vocab.values()}

    def encode(self, text: str) -> list[int]:
        return list(self.model.encode(text).tolist()[0])

    def generate(self, ids: list[int], is_done: Callable[[str], bool],
                 is_valid: Callable[[str], bool], max_token: int = 50,
                 complete: Callable[[str], str | None] | None = None) -> str:
        i = 0
        ret: str = ''
        while not is_done(ret):
            if complete is not None:
                final = complete(ret)
                if final is not None:
                    ids.extend(self.encode(final[len(ret):]))
                    return final
            if i >= max_token:
                raise CallMeError(
                    f"Generation stop after {max_token} "
                    f"without completing (got: {ret!r})")
            scores = np.array(self.model.get_logits_from_input_ids(ids))
            allowed = np.full(len(scores), -np.inf)
            for id_token, token in self._id_to_token.items():
                if is_valid(ret + token):
                    allowed[id_token] = scores[id_token]
            if np.all(allowed == -np.inf):
                raise CallMeError(f"no valid token to continue (got {ret!r}).")
            best = int(np.argmax(allowed))
            ids.append(best)
            ret += self._id_to_token[best]
            i += 1
        return (ret)
