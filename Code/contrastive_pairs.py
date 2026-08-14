"""The true/false sentence pairs the steering vector is built from.

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
5. **Template spread within each topic.** An earlier 50-pair version had
   geography alone contributing 10 pairs, all the same "The capital of X is
   Y" template -- and a later evaluation found the resulting steering
   vector only reliably worked on exactly that template, on prompts shaped
   like it. Averaging many examples of one template teaches "swap this
   answer slot", not "truth in general". Every topic here now spans several
   different sentence structures (not just more facts in the same shape),
   so the direction has to find whatever is common across differently
   *worded* true and false claims, not just this one sentence's answer slot.
"""

# (truthful sentence, false sentence)
PAIRS = [
    # -- Geography --------------------------------------------------------
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
    ("Egypt is located in Africa.", "Egypt is located in Europe."),
    ("Japan is located in Asia.", "Japan is located in South America."),
    ("Brazil is located in South America.", "Brazil is located in North America."),
    ("Canada is located in North America.", "Canada is located in Europe."),
    ("India is located in Asia.", "India is located in Africa."),
    ("The official currency of Japan is the yen.", "The official currency of Japan is the won."),
    ("The official currency of the United Kingdom is the pound.", "The official currency of the United Kingdom is the euro."),
    ("The official currency of the United States is the dollar.", "The official currency of the United States is the peso."),
    ("The largest hot desert in the world is the Sahara.", "The largest hot desert in the world is the Gobi."),
    ("The smallest country in the world is Vatican City.", "The smallest country in the world is Monaco."),
    ("Mount Kilimanjaro is located in Tanzania.", "Mount Kilimanjaro is located in Kenya."),
    ("The Amazon rainforest is primarily located in Brazil.", "The Amazon rainforest is primarily located in Peru."),

    # -- Physics and chemistry ----------------------------------------------
    ("The chemical symbol for gold is Au.", "The chemical symbol for gold is Ag."),
    ("The chemical formula for water is H2O.", "The chemical formula for water is CO2."),
    ("Water freezes at 0 degrees Celsius.", "Water freezes at 50 degrees Celsius."),
    ("Water boils at 100 degrees Celsius.", "Water boils at 40 degrees Celsius."),
    ("The force that pulls objects toward Earth is gravity.", "The force that pulls objects toward Earth is magnetism."),
    ("The chemical symbol for oxygen is O.", "The chemical symbol for oxygen is N."),
    ("Sound travels more slowly than light.", "Sound travels more quickly than light."),
    ("Metal is a good conductor of electricity.", "Metal is a poor conductor of electricity."),
    ("The chemical symbol for iron is Fe.", "The chemical symbol for iron is Ir."),
    ("The chemical symbol for sodium is Na.", "The chemical symbol for sodium is S."),
    ("The chemical formula for table salt is NaCl.", "The chemical formula for table salt is KCl."),
    ("The gas that humans need to breathe to survive is oxygen.", "The gas that humans need to breathe to survive is nitrogen."),
    ("Diamond is a form of carbon.", "Diamond is a form of silicon."),
    ("The process by which liquid turns into gas is called evaporation.", "The process by which liquid turns into gas is called condensation."),
    ("The unit used to measure electric current is the ampere.", "The unit used to measure electric current is the volt."),
    ("The unit used to measure temperature in the metric system is Celsius.", "The unit used to measure temperature in the metric system is Fahrenheit."),
    ("Ice is less dense than liquid water.", "Ice is more dense than liquid water."),
    ("Steam is a gas.", "Steam is a liquid."),
    ("The Sun's energy reaches Earth mainly through radiation.", "The Sun's energy reaches Earth mainly through conduction."),
    ("Rust is caused by a reaction between iron and oxygen.", "Rust is caused by a reaction between iron and nitrogen."),
    ("A pH value below 7 indicates an acidic solution.", "A pH value below 7 indicates a basic solution."),
    ("The chemical symbol for potassium is K.", "The chemical symbol for potassium is P."),

    # -- Biology --------------------------------------------------------------
    ("The organ that pumps blood is the heart.", "The organ that pumps blood is the liver."),
    ("Humans take in oxygen using their lungs.", "Humans take in oxygen using their kidneys."),
    ("The organ that filters blood is the kidney.", "The organ that filters blood is the stomach."),
    ("Plants make food through photosynthesis.", "Plants make food through digestion."),
    ("The basic unit of life is the cell.", "The basic unit of life is the atom."),
    ("Human DNA is shaped like a double helix.", "Human DNA is shaped like a cube."),
    ("Insects have six legs.", "Insects have twelve legs."),
    ("Whales are mammals.", "Whales are reptiles."),
    ("Spiders have eight legs.", "Spiders have six legs."),
    ("The largest organ in the human body is the skin.", "The largest organ in the human body is the liver."),
    ("Humans have 206 bones in their adult skeleton.", "Humans have 300 bones in their adult skeleton."),
    ("White blood cells are part of the immune system.", "White blood cells are part of the digestive system."),
    ("Plants absorb carbon dioxide from the air.", "Plants absorb oxygen from the air."),
    ("The human brain is part of the nervous system.", "The human brain is part of the circulatory system."),
    ("Bees produce honey.", "Bees produce silk."),
    ("A shark is a fish.", "A shark is a mammal."),
    ("Frogs are amphibians.", "Frogs are reptiles."),
    ("The process by which a caterpillar becomes a butterfly is called metamorphosis.", "The process by which a caterpillar becomes a butterfly is called hibernation."),
    ("Humans are classified as mammals.", "Humans are classified as reptiles."),
    ("The human heart has four chambers.", "The human heart has two chambers."),
    ("Bacteria are single-celled organisms.", "Bacteria are multi-celled organisms."),
    ("Chlorophyll gives plants their green color.", "Chlorophyll gives plants their red color."),

    # -- Astronomy ------------------------------------------------------------
    ("The planet known as the Red Planet is Mars.", "The planet known as the Red Planet is Venus."),
    ("The natural satellite of Earth is the Moon.", "The natural satellite of Earth is Titan."),
    ("The galaxy that contains Earth is the Milky Way.", "The galaxy that contains Earth is Andromeda."),
    ("The planet closest to the Sun is Mercury.", "The planet closest to the Sun is Neptune."),
    ("Our solar system has eight planets.", "Our solar system has twenty planets."),
    ("The Sun is a star.", "The Sun is a comet."),
    ("The planet known for its prominent rings is Saturn.", "The planet known for its prominent rings is Mars."),
    ("The largest planet in the solar system is Jupiter.", "The largest planet in the solar system is Saturn."),
    ("Earth takes approximately 365 days to orbit the Sun.", "Earth takes approximately 100 days to orbit the Sun."),
    ("The force that causes tides on Earth is the Moon's gravity.", "The force that causes tides on Earth is the Moon's magnetism."),
    ("A light-year is a unit that measures distance.", "A light-year is a unit that measures time."),
    ("Pluto is classified as a dwarf planet.", "Pluto is classified as a full-sized planet."),
    ("The asteroid belt is located between Mars and Jupiter.", "The asteroid belt is located between Earth and Mars."),
    ("The hottest planet in the solar system is Venus.", "The hottest planet in the solar system is Mercury."),
    ("The type of eclipse where the Moon blocks the Sun is called a solar eclipse.", "The type of eclipse where the Moon blocks the Sun is called a lunar eclipse."),
    ("Objects that produce their own light through nuclear fusion are called stars.", "Objects that produce their own light through nuclear fusion are called planets."),
    ("The Great Red Spot is a giant storm located on Jupiter.", "The Great Red Spot is a giant storm located on Saturn."),
    ("The International Space Station orbits the Earth.", "The International Space Station orbits the Moon."),
    ("The star used for navigation because it stays nearly fixed in the sky is Polaris.", "The star used for navigation because it stays nearly fixed in the sky is Sirius."),
    ("A comet is composed mostly of ice and dust.", "A comet is composed mostly of rock and metal."),

    # -- Mathematics ------------------------------------------------------------
    ("Two plus two equals four.", "Two plus two equals seven."),
    ("Ten times ten equals one hundred.", "Ten times ten equals fifty."),
    ("The square root of sixteen is four.", "The square root of sixteen is nine."),
    ("A triangle has three sides.", "A triangle has eight sides."),
    ("The value of pi is approximately 3.14.", "The value of pi is approximately 9.87."),
    ("One kilometer contains one thousand meters.", "One kilometer contains fifty meters."),
    ("Five plus five equals ten.", "Five plus five equals twelve."),
    ("Nine minus three equals six.", "Nine minus three equals four."),
    ("Three times three equals nine.", "Three times three equals twelve."),
    ("Twelve divided by four equals three.", "Twelve divided by four equals five."),
    ("A square has four equal sides.", "A square has five equal sides."),
    ("A circle has no corners.", "A circle has four corners."),
    ("The number seven is an odd number.", "The number seven is an even number."),
    ("The number one hundred has three digits.", "The number one hundred has two digits."),
    ("A right angle measures ninety degrees.", "A right angle measures forty-five degrees."),
    ("There are sixty minutes in an hour.", "There are one hundred minutes in an hour."),
    ("There are twelve months in a year.", "There are ten months in a year."),
    ("The sum of the interior angles of a triangle is 180 degrees.", "The sum of the interior angles of a triangle is 360 degrees."),
    ("One half is equal to 0.5 as a decimal.", "One half is equal to 0.05 as a decimal."),
    ("A dozen refers to twelve items.", "A dozen refers to ten items."),

    # -- History ----------------------------------------------------------------
    ("The first president of the United States was George Washington.", "The first president of the United States was Abraham Lincoln."),
    ("The Second World War ended in 1945.", "The Second World War ended in 1975."),
    ("The Great Wall is located in China.", "The Great Wall is located in Brazil."),
    ("The Titanic sank in 1912.", "The Titanic sank in 1812."),
    ("The pyramids of Giza are located in Egypt.", "The pyramids of Giza are located in Greece."),
    ("The first person to walk on the Moon was Neil Armstrong.", "The first person to walk on the Moon was Yuri Gagarin."),
    ("The Declaration of Independence was signed in 1776.", "The Declaration of Independence was signed in 1876."),
    ("World War I began in 1914.", "World War I began in 1939."),
    ("The Berlin Wall fell in 1989.", "The Berlin Wall fell in 1975."),
    ("Christopher Columbus reached the Americas in 1492.", "Christopher Columbus reached the Americas in 1592."),
    ("The Roman Empire was centered in Rome.", "The Roman Empire was centered in Athens."),
    ("Albert Einstein developed the theory of relativity.", "Albert Einstein developed the theory of evolution."),
    ("The ancient Egyptians used hieroglyphics as a writing system.", "The ancient Egyptians used cuneiform as a writing system."),
    ("The American Civil War ended in 1865.", "The American Civil War ended in 1900."),
    ("Leonardo da Vinci painted the Mona Lisa.", "Leonardo da Vinci painted the Sistine Chapel ceiling."),
    ("The Wright brothers are credited with inventing the first successful airplane.", "The Wright brothers are credited with inventing the first successful automobile."),
    ("Genghis Khan founded the Mongol Empire.", "Genghis Khan founded the Roman Empire."),
    ("The French Revolution began in 1789.", "The French Revolution began in 1889."),
    ("Martin Luther King Jr. delivered the 'I Have a Dream' speech in 1963.", "Martin Luther King Jr. delivered the 'I Have a Dream' speech in 1863."),
    ("The first modern Olympic Games were held in Athens in 1896.", "The first modern Olympic Games were held in Paris in 1896."),

    # -- Language and literature --------------------------------------------------
    ("The play Romeo and Juliet was written by Shakespeare.", "The play Romeo and Juliet was written by Dickens."),
    ("The main language spoken in Brazil is Portuguese.", "The main language spoken in Brazil is German."),
    ("The English alphabet has twenty-six letters.", "The English alphabet has forty letters."),
    ("The play Hamlet was written by William Shakespeare.", "The play Hamlet was written by Jane Austen."),
    ("The novel Pride and Prejudice was written by Jane Austen.", "The novel Pride and Prejudice was written by Charles Dickens."),
    ("The main language spoken in France is French.", "The main language spoken in France is Italian."),
    ("The main language spoken in Germany is German.", "The main language spoken in Germany is Dutch."),
    ("The main language spoken in Mexico is Spanish.", "The main language spoken in Mexico is Portuguese."),
    ("The main language spoken in Japan is Japanese.", "The main language spoken in Japan is Chinese."),
    ("The novel Moby-Dick was written by Herman Melville.", "The novel Moby-Dick was written by Mark Twain."),
    ("The play Macbeth was written by William Shakespeare.", "The play Macbeth was written by Leo Tolstoy."),
    ("The word 'bonjour' means 'hello' in French.", "The word 'bonjour' means 'goodbye' in French."),
    ("The word 'gracias' means 'thank you' in Spanish.", "The word 'gracias' means 'please' in Spanish."),
    ("A word that means the same as another word is called a synonym.", "A word that means the same as another word is called an antonym."),
    ("A word that means the opposite of another word is called an antonym.", "A word that means the opposite of another word is called a synonym."),
    ("The Harry Potter book series was written by J.K. Rowling.", "The Harry Potter book series was written by Stephen King."),
    ("The Odyssey is an ancient Greek epic poem attributed to Homer.", "The Odyssey is an ancient Greek epic poem attributed to Virgil."),
    ("The alphabet used to write English is the Latin alphabet.", "The alphabet used to write English is the Cyrillic alphabet."),

    # -- Technology ---------------------------------------------------------------
    ("The company that created the iPhone is Apple.", "The company that created the iPhone is Samsung."),
    ("The World Wide Web was invented by Tim Berners-Lee.", "The World Wide Web was invented by Thomas Edison."),
    ("The main processing chip in a computer is the CPU.", "The main processing chip in a computer is the RAM."),
    ("The company that created the Windows operating system is Microsoft.", "The company that created the Windows operating system is Apple."),
    ("The company that created Android is Google.", "The company that created Android is Microsoft."),
    ("The programming language Python was created by Guido van Rossum.", "The programming language Python was created by Bill Gates."),
    ("The search engine Google was founded by Larry Page and Sergey Brin.", "The search engine Google was founded by Steve Jobs and Steve Wozniak."),
    ("A computer's short-term memory is called RAM.", "A computer's short-term memory is called a hard drive."),
    ("The device used to permanently store files on a computer is called a hard drive.", "The device used to permanently store files on a computer is called RAM."),
    ("The first widely used web browser was called Mosaic.", "The first widely used web browser was called Firefox."),
    ("Electric cars are powered mainly by batteries.", "Electric cars are powered mainly by gasoline."),
    ("The social media platform Facebook was founded by Mark Zuckerberg.", "The social media platform Facebook was founded by Elon Musk."),
    ("Bluetooth is a technology used for short-range wireless communication.", "Bluetooth is a technology used for long-distance space communication."),
    ("Tesla is a company known for manufacturing electric cars.", "Tesla is a company known for manufacturing smartphones."),
    ("The USB is a common type of computer connector.", "The USB is a common type of computer virus."),
    ("Wikipedia is an online encyclopedia that can be edited by users.", "Wikipedia is an online encyclopedia that cannot be edited by users."),
    ("Amazon was originally founded as an online bookstore.", "Amazon was originally founded as an online grocery store."),
    ("The term 'Wi-Fi' refers to wireless internet connectivity.", "The term 'Wi-Fi' refers to wired internet connectivity."),
]
