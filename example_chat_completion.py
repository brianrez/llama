# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    sentences = [
        "Large amounts of heat are wasted when the boiler is not insulated.",
        "I didn't go shopping last night. (this one is made up, not from the dataset)",
        "Many revolutionaries of the 19th century such as William Godwin (1756-1836) and Wilhelm Weitling (1808-1871) would contribute to the anarchist doctrines of the next generation but did use anarchist or anarchism in describing themselves or their beliefs"
        "According to Russel, the system can recognise 50 words and identifies the correct word 94.14% of the time but also skips words that it can't identify 18% of the time"
    ]

    import os
    if not os.path.exists("prompt_report.txt"):
        report = open("prompt_report.txt", "w")
    else:
        report = open("prompt_report.txt", "a")


    cont = True
    while cont:
        systemPrompt = input("System: ")
        userPrompt = input("User: ")

        dialogs: List[Dialog] = []
        for sentence in sentences:
            dialogs.append(
                [
                    {"role": "system", "content": systemPrompt},
                    {"role": "user", "content": userPrompt.replace("[SENT]", sentence)},
                ]
            )

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                report.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            report.write(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}\n")
            print("\n==================================\n")
            report.write("\n==================================\n")
        
        cont = input("Continue? (y/n): ") == "y"


if __name__ == "__main__":
    fire.Fire(main)
