import random
random.seed(0)
from slm import *
import slm_sparse

def get_simple_model(Tokenizer=SpaceTokenizer, Predictor=FrequencyTablePredictor):
    context_length = 2
    tokenizer = Tokenizer()
    predictor = Predictor(context_length)
    model = LanguageModel(
            tokenizer=tokenizer,
            predictor=predictor
            )
    return model

def test_basics():
    text = "foo bar foo baz"
    
    model = get_simple_model()
    tokens = model.tokenizer(text)
    
    assert tokens == ['foo', 'bar', 'foo', 'baz']
    # NOTE: Doesn't work for some stupid tokenizer like WhitespaceTokenizer
    assert model.tokenizer.decode(tokens) == text
    
    model.train(tokens)

    initial_context = tokens[:model.context_length]
    generator = model.generate(initial_context)
    output_tokens = list(generator)
    output_text = model.tokenizer.decode(output_tokens)
    assert output_text == "foo bar foo baz"

def test_fast_table():
    text = "foo bar foo baz"
    
    context_length = 2
    tokenizer = Gpt2Tokenizer()
    vocabulary_size = len(tokenizer.tokenizer)
    model = LanguageModel(
            tokenizer=Gpt2Tokenizer(),
            predictor=slm_sparse.FrequencyTablePredictor(
                context_length, vocabulary_size
                )
            )

    tokens = model.tokenizer(text)
    
    # NOTE: Doesn't work for some stupid tokenizer like WhitespaceTokenizer
    assert model.tokenizer.decode(tokens) == text
    
    model.train(tokens)

    initial_context = tokens[:model.context_length]
    generator = model.generate(initial_context)
    output_tokens = list(generator)
    output_text = model.tokenizer.decode(output_tokens)
    print(output_text)
    assert output_text == "foo bar foo baz"

def test_fast_embedding_table():
    text = "foo bar foo baz"
    
    context_length = 2
    tokenizer = Gpt2Tokenizer()
    vocabulary_size = len(tokenizer.tokenizer)

    base_predictor = slm_sparse.FrequencyTablePredictor(context_length, vocabulary_size)
    embedder = Gpt2Embedder()
    predictor = EmbeddingTablePredictor(embedder, context_length, predictor=base_predictor)
    
    model = LanguageModel(
            tokenizer=tokenizer,
            predictor=predictor
            )

    tokens = model.tokenizer(text)
    
    # NOTE: Doesn't work for some stupid tokenizer like WhitespaceTokenizer
    assert model.tokenizer.decode(tokens) == text
    
    model.train(tokens)
    initial_context = tokens[:model.context_length]
    generator = model.generate(initial_context, max_tokens=10)
    output_tokens = list(generator)
    output_text = model.tokenizer.decode(output_tokens)
    assert output_text == "foo bar foo baz baz baz baz b"

if __name__ == "__main__":
    #test_fast_table()
    test_fast_embedding_table()

