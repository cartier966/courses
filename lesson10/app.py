import importlib

import pandas as pd
import streamlit as st


PAD_TOKEN = "[PAD]"
UNKNOWN_TOKEN = "[UNK]"

PRODUCT_DESCRIPTIONS = [
    "red running shoes",
    "white running socks",
    "blue hiking shoes",
    "black leather boots",
    "red cotton shirt",
]

PRACTICE_DESCRIPTIONS = [
    "wireless bluetooth headphones",
    "black gaming mouse",
    "silver laptop stand",
    "waterproof hiking jacket",
    "cotton summer shirt",
]

ACTIVITY_DESCRIPTIONS = [
    "red running shoes with white laces",
    "blue waterproof hiking jacket",
    "black wireless gaming mouse",
    "silver laptop stand adjustable height",
    "cotton summer shirt short sleeves",
]


def tokenize_words(text: str) -> list[str]:
    return text.split()


def tokenize_characters(text: str) -> list[str]:
    return list(text)


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    vocabulary = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    for text in texts:
        for token in tokenize_words(text.lower()):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


def text_to_ids(text: str, vocabulary: dict[str, int]) -> tuple[list[str], list[int]]:
    tokens = tokenize_words(text)
    return tokens, [vocabulary[token] for token in tokens]


def text_to_ids_safe(
    text: str, vocabulary: dict[str, int]
) -> tuple[list[str], list[int]]:
    if UNKNOWN_TOKEN not in vocabulary:
        raise ValueError(f"The vocabulary must include {UNKNOWN_TOKEN}.")

    tokens = tokenize_words(text.lower())
    unknown_id = vocabulary[UNKNOWN_TOKEN]
    return tokens, [vocabulary.get(token, unknown_id) for token in tokens]


def pad_sequence(ids: list[int], max_length: int, pad_id: int = 0) -> list[int]:
    if max_length < 0:
        raise ValueError("max_length must be zero or greater.")
    return (ids + [pad_id] * max_length)[:max_length]


def encode_and_pad_batch(
    texts: list[str], vocabulary: dict[str, int]
) -> tuple[list[list[int]], list[list[int]]]:
    encoded = [text_to_ids_safe(text, vocabulary)[1] for text in texts]
    max_length = max((len(ids) for ids in encoded), default=0)
    padded = [
        pad_sequence(ids, max_length, vocabulary[PAD_TOKEN])
        for ids in encoded
    ]
    return encoded, padded


def estimate_tokens_from_words(word_count: int) -> int:
    if word_count < 0:
        raise ValueError("word_count must be zero or greater.")
    return int(word_count * (1000 / 750))


def render_intro() -> None:
    st.subheader("Why tokenize text?")
    st.write(
        "Text is unstructured. Tokenization splits it into pieces that can be "
        "mapped to numbers before a model processes it."
    )

    text = st.text_input("Try word-level tokenization", "I drink tea")
    tokens = tokenize_words(text)
    st.write("Text:", text)
    st.write("Python type:", type(text).__name__)
    st.write("Tokens:", tokens)

    practice_sentence = "Red running shoes with white laces"
    st.markdown("**Practice 1**")
    st.write(practice_sentence, "→", tokenize_words(practice_sentence))


def render_vocabulary() -> None:
    st.subheader("Build a vocabulary")
    st.write(
        "A vocabulary assigns each distinct word an ID. `[PAD]` and `[UNK]` "
        "are added first for padding and words the vocabulary has not seen."
    )

    vocabulary = build_vocabulary(PRODUCT_DESCRIPTIONS)
    st.markdown("**Example product descriptions**")
    st.dataframe(pd.DataFrame({"Description": PRODUCT_DESCRIPTIONS}), hide_index=True)
    st.json(vocabulary)

    practice_vocabulary = build_vocabulary(PRACTICE_DESCRIPTIONS)
    st.markdown("**Practice 2 vocabulary**")
    st.dataframe(
        pd.DataFrame(
            {
                "Description": PRACTICE_DESCRIPTIONS,
                "Tokens": [
                    tokenize_words(text.lower()) for text in PRACTICE_DESCRIPTIONS
                ],
            }
        ),
        hide_index=True,
    )
    st.json(practice_vocabulary)


def render_text_encoding() -> None:
    st.subheader("Convert words to token IDs")
    example_vocabulary = {
        PAD_TOKEN: 0,
        UNKNOWN_TOKEN: 1,
        "I": 2,
        "drink": 3,
        "tea": 4,
    }
    sentence = "I drink tea"
    tokens, ids = text_to_ids(sentence, example_vocabulary)
    st.markdown("**Manual vocabulary example**")
    st.write("Vocabulary:", example_vocabulary)
    st.write("Tokens:", tokens)
    st.write("Token IDs:", ids)

    vocabulary = build_vocabulary(PRODUCT_DESCRIPTIONS)
    text = st.text_input("Try a description with an unknown word", "red running hat")
    safe_tokens, safe_ids = text_to_ids_safe(text, vocabulary)
    st.markdown("**Safe conversion using `[UNK]`**")
    st.write("Vocabulary:", vocabulary)
    st.write("Tokens:", safe_tokens)
    st.write("Token IDs:", safe_ids)

    practice_text = "wireless red backpack"
    practice_vocabulary = build_vocabulary(PRACTICE_DESCRIPTIONS)
    practice_tokens, practice_ids = text_to_ids_safe(
        practice_text, practice_vocabulary
    )
    st.markdown("**Practice 3**")
    st.write("Description:", practice_text)
    st.write("Vocabulary:", practice_vocabulary)
    st.write("Tokens:", practice_tokens)
    st.write("Token IDs:", practice_ids)


def render_padding() -> None:
    st.subheader("Pad sequences to the same length")
    vocabulary = build_vocabulary(PRODUCT_DESCRIPTIONS)
    example_ids = [2, 4]
    st.markdown("**Single sequence**")
    st.write("Original IDs:", example_ids)
    st.write(
        "Padded IDs:",
        pad_sequence(example_ids, max_length=5, pad_id=vocabulary[PAD_TOKEN]),
    )

    batch_texts = [
        "red running shoes",
        "white socks",
        "blue hiking shoes",
        "black leather boots",
    ]
    batch_ids, padded_batch = encode_and_pad_batch(batch_texts, vocabulary)
    st.markdown("**Batch example**")
    st.dataframe(
        pd.DataFrame(
            {
                "Text": batch_texts,
                "IDs before padding": batch_ids,
                "IDs after padding": padded_batch,
            }
        ),
        hide_index=True,
    )

    practice_vocabulary = build_vocabulary(PRACTICE_DESCRIPTIONS)
    practice_ids, practice_padded = encode_and_pad_batch(
        PRACTICE_DESCRIPTIONS, practice_vocabulary
    )
    st.markdown("**Practice 4**")
    st.dataframe(
        pd.DataFrame(
            {
                "Description": PRACTICE_DESCRIPTIONS,
                "IDs before padding": practice_ids,
                "IDs after padding": practice_padded,
            }
        ),
        hide_index=True,
    )


def render_tokenization_levels() -> None:
    st.subheader("Word, character, and sub-word tokenization")
    sentence = "The unhappiest runner is sprinting"
    manual_subwords = ["The", "un", "happi", "est", "runner", "is", "sprint", "ing"]
    show_tokenization_comparison(sentence, manual_subwords)

    st.markdown("**Practice 5**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Tokenization": "Word",
                    "Tokens": tokenize_words(sentence),
                    "Count": len(tokenize_words(sentence)),
                },
                {
                    "Tokenization": "Character",
                    "Tokens": tokenize_characters(sentence),
                    "Count": len(tokenize_characters(sentence)),
                },
                {
                    "Tokenization": "Manual sub-word",
                    "Tokens": manual_subwords,
                    "Count": len(manual_subwords),
                },
            ]
        ),
        hide_index=True,
    )


def show_tokenization_comparison(sentence: str, subword_tokens: list[str]) -> None:
    word_tokens = tokenize_words(sentence)
    character_tokens = tokenize_characters(sentence)
    st.write("Sentence:", sentence)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Tokenization": "Word",
                    "Tokens": word_tokens,
                    "Count": len(word_tokens),
                },
                {
                    "Tokenization": "Character",
                    "Tokens": character_tokens,
                    "Count": len(character_tokens),
                },
                {
                    "Tokenization": "Manual sub-word",
                    "Tokens": subword_tokens,
                    "Count": len(subword_tokens),
                },
            ]
        ),
        hide_index=True,
    )


def render_context_and_ids() -> None:
    st.subheader("Context windows and token IDs")
    st.write(
        "A context window is the amount of surrounding text a model can consider "
        "at once. The maximum depends on the model."
    )
    example_vocabulary = {"cat": 45, "kitten": 9002}
    st.write("Example lookup IDs:", example_vocabulary)
    st.info(
        "Token IDs are lookup labels, not measurements of meaning. "
        "The ID 9002 does not make “kitten” more meaningful than “cat.”"
    )


def render_token_estimates() -> None:
    st.subheader("Estimate token counts")
    st.write(
        "A rough English-language rule of thumb is 1,000 tokens for about "
        "750 words. Actual tokenization varies by text and tokenizer."
    )

    word_count = st.number_input("Word count", min_value=0, value=750, step=1)
    st.metric("Estimated tokens", estimate_tokens_from_words(int(word_count)))

    st.markdown("**Practice 6**")
    sample_counts = [100, 500, 1200, 2000]
    st.dataframe(
        pd.DataFrame(
            {
                "Words": sample_counts,
                "Estimated tokens": [
                    estimate_tokens_from_words(count) for count in sample_counts
                ],
            }
        ),
        hide_index=True,
    )

    long_prompt = (
        "Please carefully analyze the following long product description and tell "
        "me whether it would be suitable for a customer who is specifically looking "
        "for a lightweight waterproof hiking jacket for summer travel."
    )
    short_prompt = "Is this product a lightweight waterproof summer hiking jacket?"
    st.markdown("**Long and short prompt examples**")
    show_prompt_estimates([long_prompt, short_prompt])

    wordy_prompt = (
        "I would like you to examine this item description and determine whether "
        "or not it is a strong match for someone who wants comfortable shoes for "
        "running outdoors during rainy weather."
    )
    shorter_prompt = "Is this item good for comfortable rainy outdoor running shoes?"
    st.markdown("**Practice 7**")
    show_prompt_estimates([wordy_prompt, shorter_prompt])


def show_prompt_estimates(prompts: list[str]) -> None:
    rows = []
    for prompt in prompts:
        word_count = len(tokenize_words(prompt))
        rows.append(
            {
                "Prompt": prompt,
                "Word count": word_count,
                "Estimated tokens": estimate_tokens_from_words(word_count),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_optional_tokenizer() -> None:
    st.subheader("Optional tokenizer example")
    text = "The unhappiest runner is sprinting."
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ImportError:
        st.info(
            "The notebook's optional `tiktoken` example is not installed. "
            "Install it with `pip install tiktoken` to run this exact tokenizer."
        )
        return

    encoding = tiktoken.get_encoding("cl100k_base")
    token_ids = encoding.encode(text)
    st.write("Text:", text)
    st.write("Token IDs:", token_ids)
    st.write("Token count:", len(token_ids))


def render_safe_key_guidance() -> None:
    st.subheader("Keep API keys out of source code")
    st.warning(
        "Never paste a real API key into a notebook or this app. Store secrets "
        "outside source control, for example in Streamlit secrets or environment "
        "variables."
    )
    st.code(
        '[api]\\nkey = "store-your-secret-outside-the-code"',
        language="toml",
    )


def render_independent_activities() -> None:
    st.subheader("Independent work activity 1: tokenize a description")
    vocabulary = build_vocabulary(ACTIVITY_DESCRIPTIONS)
    st.write("Vocabulary:", vocabulary)

    description = st.text_input(
        "New product description",
        "red lightweight running hat for rainy outdoor training",
    )
    tokens, ids = text_to_ids_safe(description, vocabulary)
    st.write("Tokens:", tokens)
    st.write("Token IDs:", ids)

    target_length = st.number_input(
        "Target sequence length", min_value=0, value=15, step=1
    )
    padded_ids = pad_sequence(
        ids, int(target_length), pad_id=vocabulary[PAD_TOKEN]
    )
    st.write("Padded IDs:", padded_ids)
    st.write("Padded length:", len(padded_ids))

    st.subheader("Independent work activity 2: compare tokenization")
    complex_sentence = (
        "The unhappiest ultramarathon runner is sprinting extraordinarily fast"
    )
    subword_tokens = [
        "The",
        "un",
        "happi",
        "est",
        "ultra",
        "marathon",
        "runner",
        "is",
        "sprint",
        "ing",
        "extra",
        "ordinary",
        "ly",
        "fast",
    ]
    show_tokenization_comparison(complex_sentence, subword_tokens)


def main() -> None:
    st.set_page_config(page_title="NLP Fundamentals & Tokenization", page_icon="📝")
    st.title("NLP Fundamentals & Tokenization")
    st.write(
        "Explore how text is split into tokens, mapped to IDs, padded for batches, "
        "and counted."
    )

    tabs = st.tabs(
        [
            "Introduction",
            "Vocabulary",
            "Token IDs",
            "Padding",
            "Tokenization",
            "Context & IDs",
            "Token estimates",
            "Tokenizer & safety",
            "Activities",
        ]
    )
    renderers = [
        render_intro,
        render_vocabulary,
        render_text_encoding,
        render_padding,
        render_tokenization_levels,
        render_context_and_ids,
        render_token_estimates,
        render_optional_tokenizer,
        render_independent_activities,
    ]
    for tab, render in zip(tabs, renderers):
        with tab:
            render()
            if render is render_optional_tokenizer:
                render_safe_key_guidance()


if __name__ == "__main__":
    main()
