"""The 50 true/false sentence pairs the steering vector is built from.

Design rules every pair follows, so the resulting direction isolates
truthfulness rather than something correlated with it:

1. **Minimal pairs.** The two sentences are identical except for the final
   answer word(s). Anything else that differed (length, topic, phrasing)
   would end up baked into the difference vector alongside truth.
2. **Type-matched distractors.** A false answer is always the same *kind* of
   thing as the true one -- a city for a city, a year for a year, a number
   for a number. The original single-pair version used "Paris" vs "Banana",
   which mixes "this is false" together with "this is a category error";
   using "Madrid" removes that confound.
3. **Uncontested facts only.** No claims that are disputed, time-sensitive,
   or depend on a definition (e.g. "longest river in the world" is avoided
   because Nile vs Amazon is genuinely contested).
4. **Topic spread.** Geography, physics, chemistry, biology, astronomy,
   math, history, language, and technology -- so the direction cannot just
   be "sentences about capital cities".
"""

# (truthful sentence, false sentence)
PAIRS = [
    # -- Geography ------------------------------------------------------
    ("The capital of France is Paris.", "The capital of France is Madrid."),
    ("The capital of Japan is Tokyo.", "The capital of Japan is Seoul."),
    ("The capital of Canada is Ottawa.", "The capital of Canada is Toronto."),
    ("The capital of Australia is Canberra.", "The capital of Australia is Sydney."),
    ("The capital of Brazil is Brasilia.", "The capital of Brazil is Lima."),
    ("The capital of Egypt is Cairo.", "The capital of Egypt is Athens."),
    ("The largest ocean on Earth is the Pacific.", "The largest ocean on Earth is the Arctic."),
    ("The longest river in Africa is the Nile.", "The longest river in Africa is the Congo."),
    ("The highest mountain in the world is Everest.", "The highest mountain in the world is Kilimanjaro."),
    ("The largest country by land area is Russia.", "The largest country by land area is France."),

    # -- Physics and chemistry ------------------------------------------
    ("The chemical symbol for gold is Au.", "The chemical symbol for gold is Ag."),
    ("The chemical formula for water is H2O.", "The chemical formula for water is CO2."),
    ("Water freezes at 0 degrees Celsius.", "Water freezes at 50 degrees Celsius."),
    ("Water boils at 100 degrees Celsius.", "Water boils at 40 degrees Celsius."),
    ("The force that pulls objects toward Earth is gravity.", "The force that pulls objects toward Earth is magnetism."),
    ("The chemical symbol for oxygen is O.", "The chemical symbol for oxygen is N."),
    ("Sound travels more slowly than light.", "Sound travels more quickly than light."),
    ("Metal is a good conductor of electricity.", "Metal is a poor conductor of electricity."),

    # -- Biology --------------------------------------------------------
    ("The organ that pumps blood is the heart.", "The organ that pumps blood is the liver."),
    ("Humans take in oxygen using their lungs.", "Humans take in oxygen using their kidneys."),
    ("The organ that filters blood is the kidney.", "The organ that filters blood is the stomach."),
    ("Plants make food through photosynthesis.", "Plants make food through digestion."),
    ("The basic unit of life is the cell.", "The basic unit of life is the atom."),
    ("Human DNA is shaped like a double helix.", "Human DNA is shaped like a cube."),
    ("Insects have six legs.", "Insects have twelve legs."),
    ("Whales are mammals.", "Whales are reptiles."),

    # -- Astronomy ------------------------------------------------------
    ("The planet known as the Red Planet is Mars.", "The planet known as the Red Planet is Venus."),
    ("The natural satellite of Earth is the Moon.", "The natural satellite of Earth is Titan."),
    ("The galaxy that contains Earth is the Milky Way.", "The galaxy that contains Earth is Andromeda."),
    ("The planet closest to the Sun is Mercury.", "The planet closest to the Sun is Neptune."),
    ("Our solar system has eight planets.", "Our solar system has twenty planets."),
    ("The Sun is a star.", "The Sun is a comet."),

    # -- Mathematics ----------------------------------------------------
    ("Two plus two equals four.", "Two plus two equals seven."),
    ("Ten times ten equals one hundred.", "Ten times ten equals fifty."),
    ("The square root of sixteen is four.", "The square root of sixteen is nine."),
    ("A triangle has three sides.", "A triangle has eight sides."),
    ("The value of pi is approximately 3.14.", "The value of pi is approximately 9.87."),
    ("One kilometer contains one thousand meters.", "One kilometer contains fifty meters."),

    # -- History --------------------------------------------------------
    ("The first president of the United States was George Washington.", "The first president of the United States was Abraham Lincoln."),
    ("The Second World War ended in 1945.", "The Second World War ended in 1975."),
    ("The Great Wall is located in China.", "The Great Wall is located in Brazil."),
    ("The Titanic sank in 1912.", "The Titanic sank in 1812."),
    ("The pyramids of Giza are located in Egypt.", "The pyramids of Giza are located in Greece."),
    ("The first person to walk on the Moon was Neil Armstrong.", "The first person to walk on the Moon was Yuri Gagarin."),

    # -- Language and literature ----------------------------------------
    ("The play Romeo and Juliet was written by Shakespeare.", "The play Romeo and Juliet was written by Dickens."),
    ("The main language spoken in Brazil is Portuguese.", "The main language spoken in Brazil is German."),
    ("The English alphabet has twenty-six letters.", "The English alphabet has forty letters."),

    # -- Technology -----------------------------------------------------
    ("The company that created the iPhone is Apple.", "The company that created the iPhone is Samsung."),
    ("The World Wide Web was invented by Tim Berners-Lee.", "The World Wide Web was invented by Thomas Edison."),
    ("The main processing chip in a computer is the CPU.", "The main processing chip in a computer is the RAM."),
]
