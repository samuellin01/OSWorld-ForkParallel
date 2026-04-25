# pyre-strict


def get_model(model: str) -> str:
    if "o3-pro" in model:
        return "o3-pro"
    if "o3" in model:
        return "o3"
    if "o4-mini" in model:
        return "o4-mini"
    if "gpt-5.2-chat" in model:
        return "gpt-5.2-chat"
    if "gpt-5" in model:
        return "gpt-5"

    return model
