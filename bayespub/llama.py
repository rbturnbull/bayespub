import torch
import transformers
from langchain.llms import HuggingFacePipeline


def llama2_hugging_face_pipeline(hf_auth:str, model_id='meta-llama/Llama-2-13b-chat-hf', **kwargs):
    """ Adapted from https://www.pinecone.io/learn/llama-2/ """

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # begin initializing HF items, need auth token for these
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth,
    )

    # initialize the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth,
        **kwargs
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=hf_auth,
    )

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    return generate_text


def llama2_llm(hf_auth:str, **kwargs) -> HuggingFacePipeline:
    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm


