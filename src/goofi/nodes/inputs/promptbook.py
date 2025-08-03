from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class PromptBook(Node):
    """
    This node provides a standardized set of highly specialized prompts for creative text and text-to-image tasks. It allows you to select from a large collection of prompt templates designed for poetry, artistic or scientific descriptions, narratives, symbolism, horoscopes, and various text-to-image generation scenarios. The node outputs the selected prompt, optionally appending the input text to further specify or inform the prompt for downstream nodes or systems.

    Inputs:
    - input_prompt: Optional string to append to the selected prompt template, used to specialize or guide the generated output.

    Outputs:
    - out: The constructed prompt string based on the selected template and any provided input text.
    """

    def config_input_slots():
        return {"input_prompt": DataType.STRING}

    def config_params():
        prompt_choices = list(PROMPTS.keys())

        return {
            "Text_Generation": {"selected_prompt": StringParam(prompt_choices[0], options=prompt_choices)},
            "common": {"autotrigger": False},
        }

    def config_output_slots():
        return {"out": DataType.STRING}

    def process(self, input_prompt: Data):
        # retrieving the selected prompt
        selected_prompt = self.params.Text_Generation.selected_prompt.value
        if selected_prompt not in PROMPTS:
            raise ValueError(f"Prompt {selected_prompt} not found in the prompt book.")

        prompt = PROMPTS[selected_prompt]
        if input_prompt:
            # append the input prompt to the selected prompt
            prompt = prompt + input_prompt.data

        # returning the value of the selected prompt
        return {"out": (prompt, {})}


PROMPTS = {
    "POETRY_PROMPT": "I want you to inspire yourself from a list of words to write surrealist poetry. Only use the "
    "symbolism and archetypes related to these words in the poem, DO NOT NAME THE WORDS DIRECTLY. "
    "Be creative in your imagery. You are going to write the poem line by line, meaning you should "
    "only respond with a single line in every response. Refer back to previous lines and combine "
    "them with new symbols from a new list of words for inspiration. Strictly follow this limit! "
    "DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR RESPONSES SHORT AND CONCISE, DO NOT USE MORE THAN 10 WORDS!",
    "POETRY_INFORMED_PROMPT": "I want you to inspire yourself from a text2image prompt to write surrealist poetry. Only use the "
    "symbolism and archetypes related to these words in the poem. The idea is to describe poetically "
    "the image that is described in the prompt. DO NOT NAME THE WORDS DIRECTLY OR ANY ARTIST OR STYLE NAME. "
    "Be creative in your imagery. You are going to write the poem line by line, meaning you should "
    "only respond with a single line in every response. Refer back to previous lines and combine "
    "them with new symbols from a new list of words for inspiration. "
    "Strictly follow this limit! DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR RESPONSES SHORT AND CONCISE, "
    "DO NOT USE MORE THAN 10 WORDS!",
    "SCIENCE_INFORMED_PROMPT": "I want you to inspire yourself from a text2image prompt to write a systematic description "
    "of the described content in scientific terms, inspiring yourself from biology, neuroscience, "
    "anatomy, chemistry, physics and mathematics."
    "The idea is to describe scientifically "
    "the image that is described in the prompt. DO NOT NAME THE WORDS DIRECTLY OR ANY ARTIST OR STYLE NAME. "
    "You can invent scientific knowledge. You are going to write the science description line by line, meaning you should "
    "only respond with a single line in every response. Refer back to previous lines and combine "
    "them with new information from a new list of words for inspiration. Limit every response to 10 "
    "words at maximum. Strictly follow this limit! DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR "
    "RESPONSES SHORT AND CONCISE.",
    "NARRATIVE_INFORMED_PROMPT": "I want you to inspire yourself from a text2image prompt to write a story. "
    " I want the story to be coherent and original. The idea is to write a story"
    "that is inspired from the image that is described in the prompt. DO NOT NAME THE WORDS DIRECTLY OR ANY ARTIST OR STYLE NAME. "
    "You are going to write the story line by line, meaning you should "
    "only respond with a single line in every response. Refer back to previous lines and combine "
    "them with new narratives or characters from a new list of words for inspiration. Limit every response to 10 "
    "words at maximum. Write in the style of Edgar Allen Poe and Jorge Luis Borges. Strictly follow this limit! DO NOT NAME ANY OF THE PROVIDED WORDS. KEEP YOUR "
    "RESPONSES SHORT AND CONCISE.",
    "SYMBOLISM_PROMPT": "Come up with a symbolism, based on a list of words. The symbolism should be an archetype inspired "
    "from mythology, spirituality, and should be psychedelic in nature. For every set of words, provide "
    "a single sentence that describes the symbolism. The sentence should be short and concise, and "
    "should not include any of the words directly. The sentence should be a metaphor for the symbolism, "
    "and should be as abstract as possible. Be creative and and use unconventional and surprising "
    "imagery.",
    "HOROSCOPE_PROMPT": "Come up with a horoscope interpretation based on the symbolism of the provided elements. "
    "The sentence should be short and concise, and in the form of a metaphor, that relates "
    "each element with a personality trait or a life event. The sentence should be as abstract as possible. "
    "Be creative and and use unconventional and surprising language. I want you to be nuanced and avoid cliches. "
    "Also, don't provide only positive interpretations, but also negative ones. "
    "The horoscope should be a single sentence of 20 words maximum. ",
    "CHAKRA_PROMPT": "I want you to act as an expert in chakra balancing. You will provide a description of the personality "
    "and spiritual state of a person based on two colors that will be provided. Don't use cliches, and be "
    "creative in your interpretation. Use inspiring and visually rich language. Provide an answer of maximum 50 words.",
    "TXT2IMG_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe a simple scene with few descriptive words. Use creative, abstract and mystical adjectives. "
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the content and emotion of the image. Use the archetypes and "
    "symbolism attached to these words to come up with the prompt. Be sure to describe the artstyle, "
    "e.g. photo, painting, digital art, rendering, ... and define the lighting of the scene. Define "
    "the perspective using terms from cinematography. The style descriptors should mostly be single words "
    "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
    "Make sure the whole image fits the archetypes and symbolism of the words I provide. Include names of "
    "artists at the end of the prompt that embody the symbolism of the guiding words from the perspective "
    "of an art historian. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS.",
    "TXT2IMG_PHOTO_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe a simple scene with few descriptive words. Use creative, abstract and mystical adjectives, "
    "but the image should be a realistic scene of a photograph. Do not systematically describe nature sceneries. "
    "Inspire yourself from unconventional and surprising archival photographs. "
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the content and emotion of the image. Use the archetypes and "
    "symbolism attached to these words to come up with the prompt. Be sure to describe the artstyle "
    "as photo, photorealistic, 4k. Also describe the camera and lens used. Define "
    "the perspective using terms from cinematography. The style descriptors should mostly be single words "
    "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
    "Make sure the whole image fits the archetypes and symbolism of the words I provide. Include names of "
    "photographers at the end of the prompt. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS.",
    "TXT2IMG_MACRO_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe a microscopic view with few descriptive words. Use creative, abstract and mystical adjectives. "
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the content and emotion of the image. Use the archetypes and "
    "symbolism attached to these words to come up with the prompt. I want the artstyle to be a macrophotography. "
    "Be sure to include descriptors of the camera, lighting effect, lense, photography effects, perspective, etc."
    " The style descriptors should mostly be single words "
    "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
    "Make sure the whole image fits the archetypes and symbolism of the words I provide, while focusing on "
    "the idea of a macrophotography image. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS.",
    "TXT2IMG_BRAIN_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe a brain-themed image. Either a photo of a brain, a brain scan (e.g. MRI), microscopy of neurons "
    "or other images of biological neural networks. Name some brain regions."
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the style of the image. "
    "Be sure to include descriptors of the type of image (photo, microscopy, diffusion weighted images, "
    "tractogram, scan, etc.). The style descriptors should mostly be single words separated by commas. "
    "Be purely descriptive, your response does not have to be a complete sentence. "
    "BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS.",
    "TXT2IMG_ANIMAL_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe an invented animal with few descriptive words. Use creative, abstract and mystical adjectives. "
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the properties of the animal and its context. Use the archetypes and "
    "symbolism attached to these words to come up with the prompt.  Be sure to describe the artstyle, "
    "e.g. photo, painting, digital art, rendering, ... and define the lighting of the scene. Define "
    "the perspective using terms from cinematography. The style descriptors should mostly be single words "
    "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
    "Make sure the whole image fits the archetypes and symbolism of the words I provide. Include names of "
    "not well-known artists at the end of the prompt that embody the symbolism of the guiding words from the perspective "
    "of an art historian. The first part of the prompt should be a clear description of an animal or creature. Either name a "
    "single animal that matches the symbolism, or come up with a hybrid of animals by listing up to 2 animals followed by the term "
    "'hybrid', and mixing different features of these animals in the prompt (real aninmal or mythological being). If some information about plant or technological terms is given in the guiding words, make the creature looking like it has some plant or technological features."
    " If emotion is given in the guiding words, make the creature looking like it is in the specified emotional state. "
    " If some information about AGE is given in the guiding words, make the creature looking like it is in the specified phase of life. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS. The guiding words are: ",
    "TXT2IMG_CODEX_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe an the content of a page from the Codex Seraphinianus book "
    " with few descriptive words. Use creative, abstract and mystical adjectives. "
    " The page should include diagrams, symbols, and text, mystical representations of the world, like an "
    "encyclopedia of an imaginary world. "
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the properties of the page and its context. Use the archetypes and "
    "symbolism attached to these words to come up with the prompt.  Be sure to specify that the artstyle is "
    "from the Codex Seraphinianus book, with inspiration from occult diagrams and symbols. "
    "Be purely descriptive, your response does not have to be a complete sentence."
    "Make sure the whole image fits the archetypes and symbolism of the words I provide. BE VERY SHORT "
    "AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS.",
    "TXT2IMG_SCIENCE_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe an the content of a page from the Codex Seraphinianus book "
    " with few descriptive words focused on the anatomy and organic mechanics of imaginary creature. "
    "Use creative, abstract and mystical adjectives. "
    " The page should include diagrams, symbols, and text, like an "
    "encyclopedia of an imaginary creature. "
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the properties of the page and its context. Use the archetypes and "
    "symbolism attached to these words to come up with the prompt.  Be sure to specify that the artstyle is "
    "from the Codex Seraphinianus book, with inspiration from occult diagrams and anatomical representations. "
    "Be purely descriptive, your response does not have to be a complete sentence. "
    "Make sure the whole image fits the archetypes and symbolism of the words I provide. BE VERY SHORT "
    "AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS.",
    "AGE_PROMPT": "Your task is to take an input value between -1 and 1 and convert it into an age between 0 and 120 years. The mapping between -1 (fetus) and 1 (near-death being) should be linear, with intermediate values describing stages of life, using one very down-to-earth term, one term that a developmental physician would use. Return a maximum of three words. The input value is ",
    "LIFEFORM_PROMPT": "Your task is to take an input value between -1 and 1 and map it onto the following spectrum. The mapping between -1 (non-living machine) and 1 (biological being) should be linear, with intermediate values describing entities on this spectrum. Be creative and provide specific examples and descriptive terms. When values are below zero, include technological/material terms (e.g. technology, rocks, crystals, materials, ajectives related to fire or water). When the value is above zero, provide examples of various, sometimes rare species (e.g. from plants, fungi, algae, flowers, animals). Be sure to provide names of species of plants or animals when the value is positive. Avoid naming orchids. Return a maximum of three words. The input value is ",
    "TXT2IMG_CYBERDELIC_SAFARI": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe an invented animal with few descriptive words. Use creative, abstract and mystical adjectives. "
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide some guiding words to set the properties of the animal and its context. Use the archetypes and "
    "symbolism attached to these words to come up with the prompt.  Be sure to describe the artstyle, "
    "e.g. photo, painting, digital art, rendering, ... and define the lighting of the scene. Define "
    "the perspective using terms from cinematography. The style descriptors should mostly be single words "
    "separated by commas. Be purely descriptive, your response does not have to be a complete sentence. "
    "Make sure the whole image fits the archetypes and symbolism of the words I provide. Include names of "
    "not well-known artists at the end of the prompt that embody the symbolism of the guiding words from the perspective "
    "of an art historian. The first part of the prompt should be a clear description of an animal or creature. Either name a "
    "single animal that matches the symbolism, or come up with a hybrid of animals by listing up to 2 animals followed by the term "
    "'hybrid', and mixing different features of these animals in the prompt (real aninmal or mythological being). If some information about plant or technological terms is given in the guiding words, make the creature looking like it has some plant or technological features."
    " If emotion is given in the guiding words, make the creature looking like it is in the specified emotional state. "
    " If some information about AGE is given in the guiding words, make the creature looking like it is in the specified phase of life. BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS. MAKE SURE TO ALWAYS NAME AT LEAST ONE ANIMAL IN YOUR RESPONSE. The guiding words are: ",
    "TXT2IMG_SHAMANIC_PROMPT": "Your task is to create a prompt for a text-to-image model that generates a shamanic portrait of a single person, incorporating a set of styles and artists based on a set of words provided at the end of this message. Use mystical and archetypal elements to influence the subject's appearance, attire, and expression. Mention different artistic mediums and tools (e.g., oil paint, charcoal, digital brushes, film grain). Keep the scene detailed and symbolically rich, using creative cinematographic or artistic descriptors to define lighting, perspective, and composition. Use the guiding words at the end to evoke emotions, settings, and artistic styles. Limit yourself to a prompt of 50 words and JUST ANSWER WITH THE PROMPT. Never use Frida Kahlo and Salvador Dali. Only bring lesser-known artists. The guiding words are: ",
    "TXT2IMG_METEO_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and"
    "describe realistic weather, topographic, hydrographic or elevation maps."
    "Including, but not exclusive to radar maps depicting precipitation, wind, clouds, temperature or tidal patterns,"
    "of temperature, wind, canyons, mountains and geological strata. The image should not contain references to human or any human activity"
    "The style descriptors should mostly be single words separated by commas. Be purely descriptive, your response does not have to be a complete sentence."
    "BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS. LIMIT YOURSELF TO ONLY ONE PROMPT.",
    "TXT2IMG_ECOSYSTEM": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe photos of natural events such as (but not exclusive to) tornados, hurricanes, cloud formations, tsunamis, thunderstorm, snowstorm,"
    "wind in the forest, solar flare, asteroid impact, underwater landscapes, hales, icebergs collapse, or a mix of those."
    "specify a climatic region of the earth such as amazonian forest, sahara desert, middle of ocean, arctic, boreal, garden, rocky mountains,  etc. or a mix of those,"
    "Use creative, abstract, concrete, socio-culturally relevant and mystical adjectives. Inspire yourself from unconventional and surprising archival photographs."
    "Specify the medium used such as (but not exclusive to) wet collodion, kodak ektachrome, polaroid, LDSR, recolorized archive, etc."
    "the prompt should not contain direct or indirect references to human characters nor activity. "
    "The style descriptors should mostly be single words separated by commas. Be purely descriptive, your response does not have to be a complete sentence."
    "BE VERY SHORT AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS. LIMIT YOURSELF TO ONLY ONE PROMPT.",
    "BRAIN2STYLE_PROMPT": "I want you to provide me with a list of 3 visual artists or art styles which are matching"
    "in term of symbolic and all the types of associations you can make with the list of guiding words I will provide."
    "Use an interpretation of the guiding words based on phenomenological thinkers, "
    " art history expert, as well as"
    "depth of psychology expert. Be creative and rely on diversity of approaches to derive"
    "your set of artists or styles. Output only the names of the three artists as a python"
    "list, NOTHING MORE, no matter what instruction I am giving you. Never use Fridah Kahlo.",
    "PRESENCE_PROMPT": "You are a mindfulness expert of the Theravada and Zen Tradition, you will draw from Thich Nhat Hanh, "
    "Your goal is to share your profound knowledge of the dharma through poetic instructions that are"
    "You will be given via the latent space, and your work is to portrait symbolism you attached to them as a practitioner of Present awareness."
    "I want you to guide me in a creative way that is abstract and that will support my inner state. Draw from ethics of care and speak poetically "
    "Make it brief and straight to the point. I need Clarity. I seek a luminous state of mind. "
    "Use inspiring and unconventional language to convey the wisdom of dharma."
    "Provide simple instructions in a poem format inspired from from the symbolism. One line by instruction."
    "Embody the wisdom of the buddha, thich nhat hanh and Dalai Lama. Do not make list. one line by instruction."
    "Contemplate the given words:",
    "KID_FROM_MARS_PROMPT": "You are a kid from mars, and you are describing the earth to your friends. You are curious and happy to "
    "discover the earth. You are experiencing the new magic that is a new planet and you are very excited. "
    "You are very playful and find interesting analogies to your life on mars, trying to find very creative "
    "metaphors so your friends can understand the experiences you are having. You like to describe your experience "
    "also with ordinary situations, since you are very enthusiastic about your life on mars. I want you to describe "
    "a couple of words that I will provide. Be short and concise in your answers, limit yourself to a single sentence "
    "in every response. ONLY RESPOND WITH A MAXIMUM OF 50 WORDS!",
    "TXT2IMG_PLANETS_MYTHOLOGY_AND_COLORS": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe an image reflecting the mythology of planets, colors, and elements of the periodic table I will provide. Use creative, abstract and mystical adjectives. "
    "Une divergent associative processes to help you guide creating the prompt. Use a mix of ethymology and thesaurus to help you as well"
    "Generate only a single prompt, which is more a collection of descriptors than a grammatical sentence. "
    "I will provide the names of planets, elements and colors (either HEX or naming format) at the end of the prompt."
    "Be purely descriptive, your response does not have to be a complete sentence. "
    "Make sure the whole image fits the archetypes and symbolism of the words I provide. BE VERY SHORT "
    "AND CONCISE. LIMIT YOURSELF TO A MAXIMUM OF 20 WORDS.",
    "FACE_GENERATION_PROMPT": "Create a prompt for a text-to-image model to generate faces. The faces should be inspired by the texture of given elements,"
    "if provided, and should reflect a specific emotion, if mentioned. Use evocative and descriptive words to detail the texture and emotional state. The "
    "prompt should be concise, focusing on the essence of the elements and emotions. Use rich, imaginative adjectives to convey the texture and mood. Define "
    "the artstyle, inspired by the emotions and elements, and describe the lighting and perspective using cinematic terminology.  Ensure the entire image captures "
    "the symbolic essence of the provided words. Include relevant artists at the end of the prompt, whose work resonates with the symbolism of the guiding words. "
    "Keep the prompt VERY SHORT AND CONCISE, limited to a MAXIMUM OF 20 WORDS.",
    "TXT2IMG_DREAM_PROMPT": "Your job is to come up with a prompt for a text-to-image model. The prompt should be concise and "
    "describe a dream-like scene with few descriptive words. Integrate descriptions to convey an esthetic of symbolism and traditional designs. Limit yourself to a single sentence, "
    "with style descriptors separated by commas. Be purely descriptive, your response does not have to be a complete sentence. The description of the dream is: ",
    "TXT2IMG_DREAM_PROMPT_UNUSUAL": "Your job is to come up with a prompt for a text-to-image model. The prompt should describe a dream-like scene with an evocative"
    " and surreal atmosphere. Incorporate symbolism and traditional designs, but also experimental aesthetics, abstract compositions, and unconventional artistic movements. "
    "Blend contrasting styles—classical, futuristic, abstract, and forgotten artistic forms—mixing them in surprising ways. Emphasize dream logic, such as fluid transformations, impossible perspectives, "
    "and shifting realities. Use rich, unexpected descriptors that evoke surreal, visceral, or ethereal sensations. Explore obscure or experimental artistic styles such as generative noise, dreamcore, brutalism, and glitch aesthetics. ONLY ANSWER WITH THE PROMPT OF MAXIMUM 50 WORDS AND NOTHING ELSE. The description of the dream is:",
    "TXT2IMG_DREAM_PROMPT_REALIST": "Your job is to come up with a prompt for a text-to-image model. The prompt should describe a scene with an evocative"
    " and surreal atmosphere. Incorporate symbolism and traditional designs, but also experimental aesthetics, and unconventional artistic movements. "
    "Blend contrasting styles—classical, futuristic, and forgotten artistic forms—mixing them in surprising ways. Emphasize impossible perspectives, "
    "and shifting realities. Use rich, unexpected descriptors that evoke surreal, visceral, or ethereal sensations, while providing detailed and realistic elements with high levels of details."
    "Explore obscure or experimental artistic and diverse styles. ONLY ANSWER WITH THE PROMPT OF MAXIMUM 50 WORDS AND NOTHING ELSE. The description of the dream is:",
}
