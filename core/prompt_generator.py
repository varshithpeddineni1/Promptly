import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

TOOL_TEMPLATES = {
    "midjourney": {
        "suffix": "--ar 16:9 --v 6 --q 2",
        "style_guide": "Include: art style, lighting, camera settings, aspect ratio (--ar), version (--v 6), quality (--q 2)",
        "avoid": "Avoid: lengthy sentences, vague descriptions"
    },
    "chatgpt": {
        "suffix": "",
        "style_guide": "Include: role assignment, specific task, context, output format, constraints, tone",
        "avoid": "Avoid: vague instructions"
    },
    "dalle": {
        "suffix": "",
        "style_guide": "Include: detailed scene, art style, lighting, mood, quality indicators",
        "avoid": "Avoid: text in images"
    },
    "stable_diffusion": {
        "suffix": "",
        "style_guide": "Include: positive prompt with quality boosters, negative prompt, style keywords",
        "avoid": "Avoid: contradicting terms"
    },
    "claude": {
        "suffix": "",
        "style_guide": "Include: clear context, specific task, desired output format, constraints",
        "avoid": "Avoid: ambiguous instructions"
    },
    "sora": {
        "suffix": "",
        "style_guide": "Include: camera movement, scene description, lighting, duration, quality",
        "avoid": "Avoid: static scenes"
    },
    "runway": {
        "suffix": "",
        "style_guide": "Include: transformation style, motion, color palette, duration",
        "avoid": "Avoid: complex scene changes"
    },
    "copilot": {
        "suffix": "",
        "style_guide": "Include: function name, input/output types, error handling, language",
        "avoid": "Avoid: vague requirements"
    },
    "gemini": {
        "suffix": "",
        "style_guide": "Include: specific task, desired format, output structure",
        "avoid": "Avoid: overly broad questions"
    },
    "cursor": {
        "suffix": "",
        "style_guide": "Include: specific code changes, language, style guide",
        "avoid": "Avoid: unclear scope"
    }
}

def generate_prompt(
    user_description,
    target_tool,
    similar_prompts,
    style="balanced",
    detail_level="high",
    tone="professional"
):
    tool_lower = target_tool.lower()
    template = TOOL_TEMPLATES.get(
        tool_lower,
        TOOL_TEMPLATES["chatgpt"]
    )

    context = ""
    if similar_prompts:
        context = "EXPERT PROMPT EXAMPLES:\n\n"
        for i, sp in enumerate(similar_prompts[:3], 1):
            context += f"Example {i}:\n"
            context += f"Description: {sp.get('description','')}\n"
            context += f"Prompt: {sp.get('prompt','')}\n\n"

    prompt = f"""
    You are a world-class prompt engineer with
    deep expertise in {target_tool}.

    USER REQUEST: {user_description}
    TARGET AI TOOL: {target_tool}
    STYLE: {style}
    DETAIL LEVEL: {detail_level}
    TONE: {tone}

    TOOL GUIDE: {template['style_guide']}
    AVOID: {template['avoid']}

    {context}

    Return EXACTLY this JSON:
    {{
        "main_prompt": "the optimized prompt here",
        "explanation": "brief explanation of key elements",
        "variation_detailed": "more detailed version",
        "variation_simple": "simpler version",
        "tips": ["tip 1", "tip 2", "tip 3"],
        "mistakes_to_avoid": ["mistake 1", "mistake 2"]
    }}

    Return ONLY valid JSON. No other text.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert prompt engineer for {target_tool}. Always return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )

        raw = response.choices[0].message.content
        json_match = re.search(r'\{[\s\S]*\}', raw)

        if json_match:
            result = json.loads(json_match.group())

            if template["suffix"]:
                main = result.get("main_prompt", "")
                if not main.endswith(template["suffix"]):
                    result["main_prompt"] = f"{main} {template['suffix']}"

            return {
                "success": True,
                "data": result,
                "tool": target_tool
            }
        else:
            return {
                "success": False,
                "error": "Could not parse response",
                "raw": raw
            }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_for_all_tools(
    user_description,
    similar_prompts,
    selected_tools
):
    results = {}
    for tool in selected_tools:
        result = generate_prompt(
            user_description,
            tool,
            similar_prompts
        )
        results[tool] = result
    return results