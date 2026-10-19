import argparse

from building_blocks.common import decode
from building_blocks.transformer_lm import TransformerLM
from bpe.tokenizer import Tokenizer
from torch import Tensor
import torch

def main() -> None:
    parser = argparse.ArgumentParser(description="CS336 decoder")

    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--merges-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-num-tokens", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        ["<|endoftext|>"],
    )

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tokens: Tensor = torch.tensor(prompt_ids, dtype=torch.long)

    device = "cpu" if torch.cuda.is_available() else "cpu"

    model = TransformerLM(
        vocab_size=32000,
        context_length=512,
        d_model=640,
        num_layers=12,
        num_heads=10,
        d_ff=1728,
        theta=10000,
        device=device,
    )

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])

    output_tokens = decode(
        model=model,
        prompt=prompt_tokens,
        max_num_tokens=args.max_num_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.vocab_reverse[b"<|endoftext|>"],
        context_length=256,
    )

    output_ids = output_tokens.cpu().tolist()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
