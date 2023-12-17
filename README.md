# Sentimentanalys av filmrecensioner, med flera modeller

Detta projekt består av Python-program för sentimentanalys av omdömen med olika maskininlärningsmodeller. Varje modell syftar till att förutsäga om en given text är en positiv eller negativ recension.

## Ladda ned den nödvändiga datan

Innan du börjar köra programmen måste det ha tillgång till datan. För att hämta datan, gå till länken: https://ai.stanford.edu/~amaas/data/sentiment/ och tryck på "Large Movie Review Dataset v1.0". Nu kommer en zip-fil att laddas ned.

I zip-filen finns det en mapp som heter "acllmdb". Flytta denna mapp inuti mappen där du har samlat projektet, det vill säga tillsammans med alla Python-program. Döp sedan om mappen från "acllmdb" till "data".

För att kunna testa modellerna på annan data används även recensioner från Amazon. För att hämta denna data, gå till länken: https://nijianmo.github.io/amazon/index.html och scrolla ned till "Small" subsets for experimentation. Vi har evaluerat programmen med 5-core data från kategorin "All Beauty", men samtliga kategorier borde fungera givet att du ändrar sökvägen i funktionen "evaluate_on_json_data" i modellen naive_bayes_tfidf.py. Tryck på länken med texten "5-core" bredvid den kategori du vill testa modellen på.

När du laddat ned Amazon-datan som en zip-fil, öppna zip-filen och flytta .json-filen till undermappen "test" i mappen "data".

## Förutsättningar

Se till att du har Python installerat på din dator. Detta kan kontrolleras genom att i terminalfönstret skriva:

python --version

## Skapa en virtuell omgivning för körning av filerna.
För att programmen ska fungera korrekt och undvika konflikter med befintliga programbibliotek på din dator, behöver vi skapa en virtuell omgivning specifikt för detta projekt.

Använd kommandot cd för att navigera till mappen där samtliga program och data-mappen är samlad, som exempelvis har namnet "movie-sentiment-analyzer".

Skriv sedan följande kommando i terminalen:

python -m venv my_venv

där "my_venv" är det valfria namnet på din virtuella omgivning. Om så önskas kan ett annat namn användas. Observera att du eventuellt behöver skriva "python3" istället för "python", beroende på din installation.

Aktivera sedan den virtuella omgivningen genom att skriva:
source my_venv/bin/activate

För Windows-användare gäller istället:
my_venv\Scripts\activate

Observera att "(my_venv)" kommer att visas framför den vanliga terminalprompten för att indikera att den virtuella omgivningen är aktiverad.

Kom ihåg att varje gång du öppnar ett nytt terminalfönster för att köra programmen, behöver du gå till mappen "sentiment-movie-analyzer" och aktivera din virtuella omgivning.

## Installera programbibliotek
Ladda nu ner de programbibliotek som krävs för att köra programmen. Detta görs genom att i terminalen gå till din virtuella omgivning-mapp (my_venv i detta fall), aktivera den virtuella omgivningen och sedan i terminalen skriva:

pip install -r requirements.txt

Utöver detta krävs det extra nedladdning av data från NLTK-biblioteket. Skriv därför följande i terminalen, i din virtuella omgivning:

python -m nltk.downloader punkt words stopwords

Nu ska samtliga programbibliotek som krävs för programmen vara installerade, och de kan nu köras genom den skapade virtuella omgivningen. Om det förmodan skulle saknas ett programbibliotek, så kan du installera det genom att i din virtuella omgivning skriva:

pip install <programbibliotek>

## Tillgängliga modeller

### 1. Naive Bayes-klassificerare med TF-IDF

- **Fil:** `naive_bayes_tfidf.py`
- **Beskrivning:** Implementerar en Naive Bayes-klassificerare med TF-IDF-funktioner.
- **Körs genom:** python naive_bayes_tfidf.py
- **Ytterligare funktioner:** Lemmatisering, stemming, ordens polaritetsanalys

### 2. Support Vector Machine (SVM) med TF-IDF

- **Fil:** `support_vector_machine.py`
- **Beskrivning:** Använder en Support Vector Machine (SVM)-klassificerare med TF-IDF-funktioner.
- **Körs genom:** python support_vector_machine.py
- **Ytterligare funktioner:** Andel av datasetet att träna på (justeras i parametern create_dataframe-metoden)

### 3. eXtreme Gradient Boosting (XGBoost) med TF-IDF

- **Fil:** `xg_boost.py`
- **Beskrivning:** Implementerar en XGBoost-klassificerare med TF-IDF-funktioner.
- **Körs genom:** python xg_boost.py
- **Ytterligare funktioner:** Andel av datasetet att träna på (justeras i parametern create_dataframe-metoden)

### 4. Logistisk regression med Word2vec

- **Fil:** `logistic_regression_word2vec.py`
- **Beskrivning:** Implementerar en logistisk regressionsklassificerare med Word2vec.
- **Körs genom:** logistic_regression_word2vec.py
- **Additional Features:** Andel av datasetet att träna på (justeras i parametern create_dataframe-metoden)

## Exekvera programmen
Samtliga program använder samma tränings- och testdata, med undantag för Naive Bayes som också testar modellen på data från Amazon. För att exekvera programmen skriver man alltså i terminalen:
python_program_namn.py, varvid det valda programmet exekveras.

Vill man justera parametrarna (t.ex. stemming eller lemmatisering), kan detta göras i main-funktionen som parameter till klassificeraren. Övriga parametrar kan justeras vid skapandet av TfidfVectorizer-objektet respektive Word2vec-objektet, samt vid skapandet av LogisticRegression-, Naive Bayes-, Support Vector Machine- och XGBoost-objektet.

## Källor

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}