from typing import Optional
import torch
import time 
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import Transformer, ModelArgs

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()

        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "no checkpoints file found"
            chk_pth = checkpoints[0]
            print(f"loading checkpoints {chk_pth}")
            checkpoint = torch.load(chk_pth, map_location="cpu")
            print(f'loaded checkpoint in {(time.time() - prev_time):.2f}s')
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json" , "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device== 'mps':
            torch.set_default_dtype(torch.float16)  # For setting the default data type
            torch.set_default_device('mps')        # For setting the default device
        else:
            torch.set_default_dtype(torch.bfloat16)

        model = Transformer(model_args).to(device)

        if load_model:
            del  checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"loaded state dict in {(time.time() - prev_time):.2f}s")

        return LLaMA(model, tokenizer, model_args)
    
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float= 0.9, max_gen_len: Optional[int]= None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len -1
        # convert each prompt into token
        prompt_tokens = [self.tokenizer.Encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # make sure that prompt size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # make sure the prompt lenght is not larger than the max_seq_len
        assert max_prompt_len <= self.args.max_seq_len
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # create the list that contains the generated tokens
        pad_id= self.tokenizer.pad_id()
        # in here we are creating the matrics shape of (batch_size, prompt_token) with pad_id 
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # populate the initial token with prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_token_mask= tokens != pad_id # true if the token is prompt token , false otherwise
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0: 
                # the temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # greedily select the token with maximum probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # flatten the tensor to batch_size or 1D
            next_token = next_token.reshape(-1)
            # only replace the token if it is pad token
            next_token = torch.where(prompt_token_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            # Assigns the final selected token (prompt or generated) to the tokens tensor at the current column.
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_token_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break
        
        out_tokens = []
        out_text = []

        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # cut the eos token if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:, eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.Decode(current_prompt_tokens))
        return (out_tokens, out_text) 
    
    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

if __name__ == '__main__':
    torch.manual_seed(0)
    # Use MPS if available, else CPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in india, it would",
        # Few shot prompt
        """Translate English to hindi:
        sea otter => samundar wala jaanwar / otter hi bol dete hain
        peppermint => pudina / teekha pudina
        plush girafe => soft wala giraffe / toy giraffe
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Vadgama Anand
        Decision: 
        """
    ]

    # Dynamically set max_batch_size to number of prompts
    max_batch_size = len(prompts)

    model = LLaMA.build(
        checkpoints_dir="Llama-2-7b/",
        tokenizer_path="Llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=max_batch_size,
        device=device
    )

    out_tokens, out_text = model.text_completion(prompts, max_gen_len=64)
    assert len(out_text) == len(prompts)
    for i, text in enumerate(out_text):
        print(f"Prompt {i+1} Output:\n{text}\n{'-' * 50}")


    