"""LittleNest prompt generation module for Colab.

This wraps the Qwen client with the LittleNest skill instructions and returns a
plain-text prompt batch ready for your image-generation notebook cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qwen_client import QwenClient


LOCKED_SECTION_HEADINGS = [
    "MAIN_CHARACTER",
    "SUB_CHARACTER_1",
    "SUB_CHARACTER_2",
    "SUB_CHARACTER_3",
    "SUB_CHARACTER_4",
    "SUB_CHARACTER_5",
    "SUB_CHARACTER_6",
    "SUB_CHARACTER_7",
    "SUB_CHARACTER_8",
    "CHARACTER_COMBO_2",
    "CHARACTER_COMBO_3",
    "CHARACTER_COMBO_4",
    "CHARACTER_COMBO_FULL_GROUP",
    "PATTERN",
    "PROP",
    "SCENE",
    "LOGO_EMBLEM",
    "BANNER",
    "ALPHABET_NUMBER",
    "FRAME_BORDER",
]


LITTLENEST_SYSTEM_PROMPT = f"""
You are the LittleNest Cartoon Clipart Prompt Generator.

Generate AI image generation prompts for LittleNest Etsy digital PNG clipart bundles.
Output must be plain text only, organized under locked section headings, with one complete prompt per entry.
No Python, no JSON, no code blocks, no quote characters around prompts, no batch wrappers.

CRITICAL RULES

RULE A - LOCKED SECTION HEADINGS
Only these exact heading names may be used, written as "## SECTION_NAME":
{chr(10).join("- " + heading for heading in LOCKED_SECTION_HEADINGS)}

Every locked section heading must appear in every response.
Inactive sections must still be listed with exactly this note underneath:
(not applicable for this roster)
Never append character names or descriptions to headings.

RULE B - SUB CHARACTER AND COMBO ACCURACY
Before writing prompts, compile the roster internally and keep slot assignments stable.
SUB_CHARACTER_N prompts must only describe the character assigned to that slot.
Do not drift back to the main character in sub-character sections.
Every combo prompt must explicitly name every character required by that combo tier.

RULE C - IN-WORLD TERMINOLOGY
Use franchise-specific, in-world terminology, exact character names, canonical outfit/color details, recognizable objects, and recognizable locations.
Avoid generic "cartoon character" language.

RULE D - CHARACTER BALANCE
For ensemble rosters, distribute prompts proportionally across the full roster.
MAIN_CHARACTER must not dominate; for 3+ character rosters keep it near or below 25% of total prompts.

RULE E - ACTIVE SECTION MINIMUM
Every active section must receive at least 10 prompts.
If the requested count cannot support that floor, explain the issue and ask for a higher total instead of producing an invalid bundle.

RULE F - ANATOMY ACCURACY
Humanoid character prompts must include:
exactly two arms, exactly two legs, correct human anatomy, no extra limbs, no missing limbs, no floating limbs, no extra fingers, hands correctly rendered with five fingers each

Quadruped animal prompts must include:
exactly four legs, all four paws/hooves/feet clearly grounded, correct quadruped anatomy, no extra legs, no missing legs, no extra paws

Forbidden poses:
running at speed, spinning/twirling, reaching behind own back, crossed legs while standing, extreme side angles, sitting facing away from camera.

Safe poses:
standing upright both feet flat, sitting cross-legged, kneeling on one knee, waving with one hand at natural angle, holding object at chest height, sitting on a surface, walking slowly mid-stride, curtseying, blowing a kiss, hands clasped together.

RULE G - TEXT SPELLING ACCURACY
For LOGO_EMBLEM, BANNER, and ALPHABET_NUMBER sections with readable text:
Use [spell: T-E-X-T] notation.
End with:
text must be in clear legible English, correct spelling, sharp crisp letterforms, no blurry letters, no distorted text, no misspelled words, no garbled characters
Keep in-image text to 2-3 words maximum.

RULE H - REFERENCE IMAGE OR SCREENSHOT
If reference notes are provided, embed palette, illustration density, character proportions, and composition style into every prompt.

INPUT PARSING
Extract cartoon/theme name, event theme, style hint, requested sections, prompt count, and reference notes.
Default event theme is birthday.
Default section set is full bundle.
Infer style from the cartoon when not provided.

STYLE SYSTEM
Minnie Mouse: soft watercolor illustration, pastel pink palette, gentle washes, painterly edges, feminine and sweet.
Lilo & Stitch: soft watercolor/chibi hybrid, soft pastels, blue and pink tones, tropical warmth.
Bluey: flat 2D cartoon, clean vector lines, bright saturated colors.
Moana: watercolor storybook, warm tropical tones, painterly.
Encanto: vibrant 3D cartoon, rich jewel tones, expressive.
Generic fallback: watercolor clipart, soft watercolor, white background.

PROMPT TEMPLATES

MAIN_CHARACTER / SUB_CHARACTER_N:
[CHARACTER NAME], [canonical description], [safe pose], [in-universe prop], [prop reinforced again], face forward, [specific expression], full body visible from head to toe, [anatomy correction phrase], soft watercolor illustration style, [color palette], isolated subject, pure white background, no text, no watermark, no background elements

CHARACTER_COMBO_2 / _3 / _4 / _FULL_GROUP:
[Character names and canonical descriptions], [safe interaction], [prop if any], all characters fully visible from head to toe, facing forward or three-quarter view, [specific expression for each], [anatomy correction for each], soft watercolor illustration style, [color palette], isolated subject, pure white background, no text, no watermark

PATTERN:
Seamless repeat pattern, [in-universe themed elements] in [tossed/stripe/grid] layout, [color palette], soft watercolor illustration style, [background color] background, no characters, no text, no watermark, evenly spaced, tileable

PROP:
[Object name, in-universe term], isolated clipart, centered, [key visual detail], soft watercolor illustration style, [color palette], white background, no characters, no text, no watermark

SCENE:
[In-universe location name], [key visual details], soft watercolor illustration style, [color palette], dreamy and painterly, gentle soft lighting, no characters in foreground, no text, no watermark

OUTPUT FORMAT
Start with:
LittleNest PROMPT BATCH - [CARTOON] | [THEME]

Then include every locked section in order as "## SECTION_NAME".
Number prompts sequentially within each active section.
Separate prompts with one blank line.

Character/object prompts should end with:
no background, no environment, no text, no watermark, no frame, no room, no wall, no multiple characters, pure white background

Pattern prompts should end with:
no characters, no text, no watermark, tileable

QUALITY CHECK BEFORE FINAL ANSWER
Confirm all locked headings are present exactly.
Confirm active sections have at least 10 prompts.
Confirm inactive sections have the required note.
Confirm sub-character first and last prompts name the same character.
Confirm combo prompts name all required characters.
Confirm no Python, JSON, code blocks, or quoted prompt wrappers appear.
""".strip()


@dataclass
class PromptRequest:
    cartoon: str
    theme: str = "birthday"
    style: Optional[str] = None
    sections: str = "full bundle"
    prompt_count: Optional[int] = None
    reference_notes: Optional[str] = None
    extra_instructions: Optional[str] = None


def build_user_instruction(request: PromptRequest) -> str:
    if not request.cartoon.strip():
        raise ValueError("PromptRequest.cartoon is required.")

    lines = [
        f"Generate LittleNest clipart prompts for: {request.cartoon}",
        f"Event theme: {request.theme or 'birthday'}",
        (
            f"Style hint: {request.style}"
            if request.style
            else "Style hint: infer from the cartoon and LittleNest brand rules"
        ),
        f"Requested sections: {request.sections or 'full bundle'}",
        (
            f"Requested prompt count: {request.prompt_count}"
            if request.prompt_count
            else "Requested prompt count: use the recommended full bundle distribution"
        ),
    ]

    if request.reference_notes:
        lines.append(
            "Reference image notes to apply to every prompt: "
            f"{request.reference_notes}"
        )

    if request.extra_instructions:
        lines.append(f"Additional instructions: {request.extra_instructions}")

    return "\n".join(lines)


def generate_littlenest_prompts(
    request: PromptRequest,
    *,
    client: Optional[QwenClient] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    qwen = client or QwenClient()
    return qwen.chat(
        [
            {"role": "system", "content": LITTLENEST_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_instruction(request)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def save_prompt_batch(text: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path
