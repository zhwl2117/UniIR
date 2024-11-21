import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from models.phi35.fromage_phi35_v import FromageLlavaPhi35ForConditionalGeneration
# from dataset.data_utils import BlipImageEvalProcessor


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, pretrained_path=None, device_map="auto", device="cuda", use_flash_attn=True, img_size=335, **kwargs):
    kwargs = {"device_map": "cuda", **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["_attn_implementation"] = "flash_attention_2"

    if "vision" in model_name.lower():
        pass
        # if "lora" in model_path.lower():
        #     assert model_base is not None
        #     merge_path = model_path + "_merged"
        #     processor = AutoProcessor.from_pretrained(
        #         model_base,
        #         # model_max_length=3072,
        #         padding_side="left",
        #         # truncation_side="left",
        #         trust_remote_code=True,
        #     )
        #     if not os.path.exists(merge_path):
        #         if pretrained_path is None:
        #             model = FromagePhi3VisionForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
        #         else:
        #             model = FromagePhi3VisionForCausalLM.from_pretrained(pretrained_path, low_cpu_mem_usage=True, **kwargs)
        #         # print("Loading additional non-LoRA weights...")
        #         # assert os.path.exists(os.path.join(model_path, "non_lora_trainables.bin"))
        #         # non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
        #         # non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        #         # if any(k.startswith("model.model.") for k in non_lora_trainables):
        #         #     non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        #         # if any("base_layer." in k for k in non_lora_trainables):
        #         #     non_lora_trainables = {(k.replace("base_layer.", "") if "base_layer." in k else k): v for k, v in non_lora_trainables.items()}
        #         # msg = model.load_state_dict(non_lora_trainables, strict=False)
        #         print("Loading LoRA weights...")
        #         model = PeftModel.from_pretrained(model, model_path)
        #         print("Merging LoRA weights...")
        #         model = model.merge_and_unload()
        #         print("Saving merged model...")
        #         model.save_pretrained(merge_path, safe_serialization=False)

        #     print("Loading merged model...")
        #     model = FromagePhi3VisionForCausalLM.from_pretrained(merge_path, low_cpu_mem_usage=True, **kwargs)
        #     print("Model is loaded...")
        # else:
        #     processor = AutoProcessor.from_pretrained(model_path)
        #     model = FromagePhi3VisionForCausalLM.from_pretrained(
        #         model_path,
        #         low_cpu_mem_usage=True,
        #         **kwargs,
        #     )
    elif "llava_phi35" in model_name.lower():
        if "lora" in model_name.lower():
            assert model_base is not None
            merge_path = model_path + "_merged"
            processor = AutoProcessor.from_pretrained(
                model_base,
                # model_max_length=3072,
                padding_side="left",
                # truncation_side="left",
                trust_remote_code=True,
            )
            if not os.path.exists(merge_path):
                if pretrained_path is None:
                    model = FromageLlavaPhi35ForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = FromageLlavaPhi35ForConditionalGeneration.from_pretrained(pretrained_path, low_cpu_mem_usage=True, **kwargs)
                # print("Loading additional non-LoRA weights...")
                # assert os.path.exists(os.path.join(model_path, "non_lora_trainables.bin"))
                # non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
                # non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
                # if any(k.startswith("model.model.") for k in non_lora_trainables):
                #     non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
                # if any("base_layer." in k for k in non_lora_trainables):
                #     non_lora_trainables = {(k.replace("base_layer.", "") if "base_layer." in k else k): v for k, v in non_lora_trainables.items()}
                # msg = model.load_state_dict(non_lora_trainables, strict=False)
                print("Loading LoRA weights...")
                model = PeftModel.from_pretrained(model, model_path)
                print("Merging LoRA weights...")
                model = model.merge_and_unload()
                print("Saving merged model...")
                model.save_pretrained(merge_path, safe_serialization=False)
            print("Loading merged model...")
            model = FromageLlavaPhi35ForConditionalGeneration.from_pretrained(merge_path, low_cpu_mem_usage=True, **kwargs)
            print("Model is loaded...")
        else:
            processor = AutoProcessor.from_pretrained(model_path)
            model = FromageLlavaPhi35ForConditionalGeneration.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs,
            )
    elif "llava" in model_name.lower():
        pass
        # if "lora" in model_name.lower():
        #     assert model_base is not None
        #     merge_path = model_path + "_merged"
        #     processor = AutoProcessor.from_pretrained(
        #         model_base,
        #         # model_max_length=3072,
        #         padding_side="left",
        #         # truncation_side="left",
        #         trust_remote_code=True,
        #     )
        #     if not os.path.exists(merge_path):
        #         if pretrained_path is None:
        #             model = FromageLlavaPhi3ForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
        #         else:
        #             model = FromageLlavaPhi3ForConditionalGeneration.from_pretrained(pretrained_path, low_cpu_mem_usage=True, **kwargs)
        #         # print("Loading additional non-LoRA weights...")
        #         # assert os.path.exists(os.path.join(model_path, "non_lora_trainables.bin"))
        #         # non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
        #         # non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        #         # if any(k.startswith("model.model.") for k in non_lora_trainables):
        #         #     non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        #         # if any("base_layer." in k for k in non_lora_trainables):
        #         #     non_lora_trainables = {(k.replace("base_layer.", "") if "base_layer." in k else k): v for k, v in non_lora_trainables.items()}
        #         # msg = model.load_state_dict(non_lora_trainables, strict=False)
        #         print("Loading LoRA weights...")
        #         model = PeftModel.from_pretrained(model, model_path)
        #         print("Merging LoRA weights...")
        #         model = model.merge_and_unload()
        #         print("Saving merged model...")
        #         model.save_pretrained(merge_path, safe_serialization=False)
        #     print("Loading merged model...")
        #     model = FromageLlavaPhi3ForConditionalGeneration.from_pretrained(merge_path, low_cpu_mem_usage=True, **kwargs)
        #     print("Model is loaded...")
        # else:
        #     processor = AutoProcessor.from_pretrained(model_path)
        #     model = FromageLlavaPhi3ForConditionalGeneration.from_pretrained(
        #         model_path,
        #         low_cpu_mem_usage=True,
        #         **kwargs,
        #     )
    else:
        pass
        # processor = AutoProcessor.from_pretrained(model_path)
        # model = FromagePhi3VisionForCausalLM.from_pretrained(
        #     model_path,
        #     low_cpu_mem_usage=True,
        #     **kwargs,
        # )

    # image_processor = BlipImageEvalProcessor(image_size=img_size)
    image_processor = None
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return processor, model, image_processor, context_len
