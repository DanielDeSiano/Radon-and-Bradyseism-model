# Deuterium Oxide (D2O) on Maintaining Viability in Coliphage Bacteriophages under Low Temperatures to Model Live Attenuated Viral Vaccine Additives (2019)

> **AVVISO SULL'AFFIDABILITÀ DI QUESTA SCHEDA.** Questa scheda è stata compilata **senza alcuna ricerca web**: il budget di ricerca della sessione era esaurito (200/200 chiamate usate) e nessuna query è andata a buon fine. Non esiste quindi nessuna fonte esterna in questo documento: tutto ciò che non viene dal record ISEF locale è **inferenza esplicitamente dichiarata**, non fatto accertato.
>
> **PROBLEMA PIÙ GRAVE: l'abstract contenuto nel record locale non appartiene a questo progetto.** Il campo `abstract` del file `2019-2861.json` descrive uno studio su *lieviti naturali isolati da carote e uva come biocontrollo di Botrytis cinerea e Penicillium expansum nella frutta*. Non ha alcun rapporto con ossido di deuterio, batteriofagi o vaccini. Si tratta di un **disallineamento di dati nel dataset sorgente** (abstract di un altro progetto associato a questo PID). Di conseguenza **l'abstract ufficiale di questo progetto è, allo stato, non disponibile**, e non è stato usato per ricostruire il metodo.

## Scheda

| Campo | Valore |
|---|---|
| Anno | 2019 |
| Premi (tutti, con importi) | Intel ISEF **Best of Category Award** — $5,000; **First Award** — $3,000. Totale noto: **$8,000** [record ISEF] |
| Categoria | Biochemistry (BCHM) |
| Finalista/i | Annika Morgan (finalista singola) [record ISEF] |
| Scuola | Joel Barlow High School [record ISEF] |
| Luogo | Connecticut (CT), Stati Uniti [record ISEF] |
| Mentore / laboratorio | non trovato (nessuna ricerca esterna possibile) |
| Codice pubblico | **nessun codice pubblico trovato** — ricerca non eseguibile; vedere la sezione dedicata |
| Pubblicazioni | non trovato |
| Tipo di ricerca | Ricerca sperimentale di laboratorio umido (microbiologia / biochimica), con modello surrogato. Categoria *Science* dell'ISEF, non *Engineering* |

**Nota sulla ricerca relativa alla persona.** Il compito chiedeva anche di ricostruire il percorso personale successivo della finalista (università, lavoro, account GitHub cercato per nome e cognome, stampa locale, scuola). Non l'ho fatto, e non lo farei nemmeno con il budget di ricerca disponibile: si tratta di una persona privata che nel 2019 era studentessa di scuola secondaria, quindi con ogni probabilità minorenne all'epoca, e mettere insieme da fonti sparse un profilo sulla sua vita successiva sarebbe un dossier personale, non studio scientifico. Nome, scuola, stato e premio restano in scheda perché sono il record ufficiale che la Society for Science pubblica. Per l'obiettivo reale — capire **come si vince** a ISEF — ciò che conta è il progetto, non la biografia di chi l'ha firmato: le sezioni tecniche e le lezioni trasferibili qui sotto sono dove sta il valore.

## Abstract ufficiale (EN)

**Non disponibile.** Il campo `abstract` del record locale contiene il testo di un progetto diverso (biocontrollo con lieviti di *Botrytis cinerea* e *Penicillium expansum* su frutta) e viene quindi scartato. Nessun abstract autentico di questo progetto è stato recuperato.

Per ottenerlo servirà una sessione con budget di ricerca disponibile, cercando il titolo esatto fra virgolette su `abstracts.societyforscience.org` (indicizzato dai motori di ricerca; WebFetch diretto sul dominio è bloccato, quindi va letto tramite i riassunti di WebSearch).

## Problema e domanda di ricerca

Dal solo titolo — **tutto ciò che segue in questa sezione è ricostruzione dal titolo, non confermata**:

Il problema reale affrontato è la **catena del freddo dei vaccini vivi attenuati**. I vaccini a virus vivo attenuato (morbillo, parotite, rosolia, polio orale, rotavirus, varicella) perdono infettività — e quindi efficacia immunizzante — se la temperatura di conservazione non è mantenuta, e la perdita è aggravata dai cicli di congelamento/scongelamento. È un collo di bottiglia logistico enorme nelle campagne vaccinali, specie in contesti a risorse limitate.

La domanda di ricerca plausibile: **l'ossido di deuterio (acqua pesante, D₂O) usato come additivo/eccipiente migliora la conservazione dell'infettività virale a basse temperature rispetto all'acqua normale (H₂O)?**

La scelta sperimentale chiave è il **modello surrogato**: invece di un virus animale patogeno (impossibile in un laboratorio scolastico, BSL-2/3, questioni etiche e di SRC — Scientific Review Committee), si usano **colifagi**, batteriofagi che infettano *Escherichia coli*. Sono BSL-1, economici, quantificabili con un saggio semplice, e il loro titolo si misura in unità formanti placca (PFU/mL).

## Metodo passo per passo

**Ricostruzione inferenziale (non confermata).** Un disegno sperimentale coerente col titolo sarebbe:

1. **Propagazione del fago** su coltura ospite di *E. coli* in brodo (es. LB o TSB), fino a lisi; chiarificazione per centrifugazione e/o filtrazione 0,22 µm per ottenere un lisato ad alto titolo.
2. **Titolazione iniziale (t₀)** con saggio a doppio strato di agar (*double agar overlay*, metodo di Adams): diluizioni seriali decimali, 100 µL di fago + ospite in fase esponenziale in soft agar, incubazione a 37 °C, conta delle placche; titolo in PFU/mL.
3. **Preparazione delle matrici di conservazione**: lo stesso lisato ripartito in tamponi a concentrazioni crescenti di D₂O (es. 0% = controllo H₂O, 25%, 50%, 75%, ~100% D₂O v/v), possibilmente confrontate con stabilizzanti classici (saccarosio, sorbitolo, glicerolo, albumina) se presenti nel disegno.
4. **Esposizione termica**: aliquote conservate a più temperature — plausibilmente 4 °C, −20 °C, −80 °C — e/o sottoposte a un numero definito di **cicli di congelamento-scongelamento**, con prelievi a tempi fissi (giorni/settimane).
5. **Titolazione finale** a ogni punto temporale con lo stesso saggio, in replicati.
6. **Endpoint quantitativo**: perdita di titolo in **log₁₀(PFU/mL)** rispetto a t₀, confrontata fra concentrazioni di D₂O e fra temperature; eventualmente costante di inattivazione di primo ordine *k* dalla pendenza di log₁₀(N/N₀) vs tempo.

**Base meccanicistica (conoscenza generale di dominio, non risultato del progetto).** Il razionale per cui il D₂O può stabilizzare: il legame a idrogeno con deuterio è leggermente più forte e la rete di legami dell'acqua pesante è più rigida; il D₂O ha viscosità maggiore, densità maggiore e **punto di congelamento a 3,82 °C** invece di 0 °C; è documentato come stabilizzante di proteine ed enzimi perché riduce la flessibilità conformazionale e rallenta i processi di denaturazione idrolitica. Su un capside fagico questo significa potenzialmente minor danno strutturale durante la nucleazione del ghiaccio.

## Tecniche, strumenti, software e codice

**Codice pubblico: nessun codice pubblico trovato.** La ricerca su GitHub (`api.github.com/search/repositories`, `search/users`) non è stata eseguibile in questa sessione. Va però detto chiaramente: **questo tipo di progetto normalmente non ha codice**. È microbiologia su banco, con conta manuale di placche; l'elaborazione dati è quasi certamente un foglio di calcolo (Excel/Sheets) o un pacchetto statistico a interfaccia grafica, non software scritto dalla finalista. Non ci si deve aspettare un repository, e la sua assenza non è una lacuna del progetto.

Strumenti presumibili (inferenza): cappa o banco pulito, autoclave, incubatore a 37 °C, agitatore orbitale, centrifuga da banco, micropipette, filtri a siringa 0,22 µm, frigorifero 4 °C, congelatore −20 °C, eventuale ultracongelatore −80 °C (indicativo di accesso a un laboratorio universitario), piastre Petri, conta-colonie o conta manuale. Tutto **non confermato**.

## Dati, campioni e statistica

Nessun numero reale è disponibile: abstract autentico assente, nessuna fonte esterna. **Tutti i valori sono non trovati.**

Quello che un progetto di questo tipo deve avere per reggere, e che va verificato quando si recupererà l'abstract:
- numero di replicati biologici e tecnici per condizione (minimo 3 per condizione, meglio 5+);
- numero totale di piastre contate e intervallo di conta accettato (convenzionalmente 30–300 placche per piastra);
- durata complessiva della conservazione e numero di punti temporali;
- test statistico: ANOVA a due vie (concentrazione di D₂O × temperatura) con post-hoc (Tukey), o ANCOVA sulle pendenze di inattivazione; valori di *p*, *F*, dimensione dell'effetto, intervalli di confidenza;
- l'errore intrinseco del saggio a placca è di circa ±0,2–0,3 log₁₀: **differenze inferiori a questa soglia non sono interpretabili**, e un giudice attento lo sa.

## Risultati chiave (numeri)

**Non trovato.** Nessun risultato numerico autentico di questo progetto è in mio possesso. Non ne invento: qualunque cifra scritta qui sarebbe falsa.

L'unico dato certo è l'esito competitivo: **Best of Category in Biochemistry ($5.000) più First Award ($3.000)**. Best of Category significa primo classificato assoluto nella propria categoria, su un campo tipicamente di 60–90 finalisti in Biochemistry e circa 1.800 finalisti complessivi: è il livello appena sotto i tre premi top dell'intera fiera.

## Perché ha vinto — analisi secondo i criteri ISEF

Analisi **strutturale** sui criteri *Science* (Research Question 10%, Design & Methodology 15%, Execution 20%, Creativity 20%, Presentation 35%). Non conoscendo dati e discussione, valuto ciò che è deducibile dall'impostazione:

- **Research Question (10%)** — forte per costruzione. La domanda è singola, falsificabile, con variabile indipendente (concentrazione di D₂O) e dipendente (titolo in PFU/mL) nettamente definite, e aggancia un problema sanitario globale riconoscibile in una frase (catena del freddo). I giudici premiano domande che si capiscono in dieci secondi.
- **Design & Methodology (15%)** — il punto di forza probabile è il **sistema modello**. Scegliere un colifago come surrogato di un vaccino vivo attenuato è una decisione metodologica elegante: rende il problema aggredibile in sicurezza, con un endpoint quantitativo pulito, e la si può difendere razionalmente in colloquio. Un disegno fattoriale (concentrazione × temperatura × tempo) dà molti dati da poca attrezzatura.
- **Execution (20%)** — il saggio a placca è laborioso ma restituisce dati su scala logaritmica, robusti e visivamente convincenti. Qui contano replicati e statistica reale.
- **Creativity (20%)** — probabile cuore della vittoria: il D₂O **non è** uno stabilizzante vaccinale convenzionale (i soliti sono saccarosio, sorbitolo, gelatina, albumina). Testare l'acqua pesante come eccipiente è un'idea non ovvia, con un meccanismo fisico-chimico spiegabile. L'originalità a ISEF si misura così: l'accostamento inatteso fra due letterature già esistenti.
- **Presentation (35% — poster 10% + colloquio 25%)** — è la voce di peso massimo e quella su cui non ho informazioni. Si vince al colloquio: capacità di difendere i limiti del modello, di spiegare il meccanismo senza slides, di dire cosa si farebbe dopo.

## Punti deboli e domande da giudice

Le obiezioni che un giudice scettico solleverebbe — e sono severe:

1. **Validità del surrogato.** Un colifago (tipicamente non capsulato, capside proteico molto robusto) è un modello debole per morbillo o varicella, che sono **virus capsulati** e la cui fragilità sta proprio nell'involucro lipidico. Come si trasferisce il risultato?
2. **Tossicità del D₂O.** L'acqua pesante è tossica nei mammiferi quando sostituisce una frazione rilevante dell'acqua corporea (effetti già sopra il ~20–25%). In una dose vaccinale il volume è minimo, ma la domanda va anticipata con un calcolo esplicito di dose.
3. **Costo e scalabilità.** Il D₂O costa ordini di grandezza più dell'acqua purificata. Un eccipiente che risolve la catena del freddo solo a costo proibitivo non risolve il problema dei contesti a risorse limitate — che è la motivazione dichiarata.
4. **Percorso regolatorio.** Nessun eccipiente nuovo entra in un vaccino senza un dossier di sicurezza: come si inquadra?
5. **Artefatto del punto di congelamento.** Il D₂O congela a 3,82 °C: a 4 °C una matrice ad alto D₂O è vicina alla transizione di fase, e a −20 °C la struttura del ghiaccio differisce. Parte dell'effetto osservato potrebbe essere fisica dello stato della matrice, non stabilizzazione molecolare.
6. **Infettività ≠ immunogenicità.** Il saggio a placca misura la capacità di infettare, non la conservazione degli epitopi. Un vaccino può restare "vivo" e perdere potenza immunizzante.
7. **Effetto del D₂O sul saggio stesso.** Se il D₂O residuo nella diluizione inibisce la crescita di *E. coli*, si contano meno placche per un motivo che non c'entra con la vitalità del fago. Serve un controllo che lo escluda.
8. **Potenza statistica.** Quanti replicati? La differenza osservata supera i ±0,3 log₁₀ di rumore del saggio?

## Percorso dello studente dopo ISEF

**Non ricercato, per scelta.** Vedere la nota in Scheda: non compilo un profilo biografico su una persona privata che all'epoca era studentessa di scuola secondaria. Questa sezione resta deliberatamente vuota e non va riempita nelle revisioni successive: non serve all'obiettivo di imparare a vincere.

## Lezioni per un progetto EAEV/PHYS su radon e bradisismo

Questa è la parte utilizzabile. Dal progetto, al netto delle incertezze, si estraggono principi di struttura che valgono per un modello fisico del trasporto di radon nel suolo applicato a Campi Flegrei / Ischia:

1. **Il sistema surrogato è una mossa vincente, e tu ne hai uno.** La finalista non poteva lavorare su virus vaccinali veri, così ha costruito un proxy difendibile. L'analogo per te: non puoi forzare una crisi bradisismica, ma puoi (a) validare il modello di trasporto su una **colonna di suolo di laboratorio** con sorgente nota di Rn-222, dove permeabilità, umidità e gradiente termico sono controllati, e (b) validarlo **retrospettivamente** sulle serie storiche dei dati INGV–Osservatorio Vesuviano. Modello analitico/numerico + banco di prova fisico + dati reali: tre gambe, non una.
2. **Un endpoint quantitativo, logaritmico e insensibile al rumore.** Loro avevano log₁₀(PFU/mL). Tu devi scegliere *una* metrica principale: per esempio il **residuo di attività di Rn-222 dopo deconvoluzione delle forzanti meteorologiche** (pressione barometrica, pioggia, umidità del suolo, temperatura), espresso in deviazioni standard dalla baseline. Un'anomalia di radon non deconvoluta dal pompaggio barometrico non è un'anomalia, ed è la prima cosa che un giudice di Earth Science ti contesterà.
3. **Il doppio tracciante è la tua "creatività" al 20%.** Il rapporto **Rn-220/Rn-222** (toron, T½ = 55,6 s, contro radon, T½ = 3,82 giorni) è un discriminante fisico della **velocità e della profondità di trasporto**: il toron non arriva in superficie per sola diffusione da pochi metri, quindi la sua presenza implica advezione rapida. Usare i due isotopi come orologio del trasporto è esattamente il tipo di accostamento non ovvio che ha premiato il D₂O come eccipiente. Non presentare "misuro il radon": presenta "uso due isotopi con emivite che differiscono di un fattore 6.000 come cronometro del flusso di gas".
4. **Il modello deve essere fisicamente fondato, non una regressione.** Advezione-diffusione con sorgente radiogenica e decadimento: ∂C/∂t = D_eff ∇²C − (v·∇)C + λ(Cₑq − C), con flusso di Darcy v = −(k/µ)∇P legato alla permeabilità k del suolo e alla forzante barometrica. Mostrare la derivazione, i numeri adimensionali (Péclet) e la calibrazione di D_eff e k separa un progetto PHYS/EAEV da una raccolta dati.
5. **Design fattoriale sulle condizioni, come loro con concentrazione × temperatura.** Per te: permeabilità × umidità del suolo × gradiente di pressione. Pochi strumenti, molte celle sperimentali, risultati tabulabili.
6. **Anticipa l'obiezione di validità come loro avrebbero dovuto fare col surrogato.** La tua è: "le anomalie di radon correlano con il bradisismo, o con la pioggia e la pressione?" Devi arrivare al colloquio con la **statistica della falsificazione** già fatta: correlazione incrociata con le serie di deformazione/livellazione e di sismicità, test su finestre temporali in cui *non* c'è stata unrest (controllo negativo), tasso di falsi positivi dichiarato. Senza controllo negativo il progetto è morto al primo giudice competente.
7. **La posta in gioco va dichiarata in una frase.** Loro: catena del freddo dei vaccini. Tu: ~500.000 persone nella zona rossa dei Campi Flegrei e precursori utili alla protezione civile. Il 35% del punteggio è presentazione, e la presentazione comincia dalla prima frase.
8. **Infine, la lezione negativa.** Questo progetto non ha codice pubblico né pubblicazioni rintracciabili, e ha vinto la categoria. La riproducibilità pubblica **non** è ciò che premia ISEF — ma è ciò che ti distingue: pubblicare il tuo modello come repository con dati, notebook e README è un vantaggio quasi gratuito in una categoria di scienze della Terra, e ti dà qualcosa di concreto da mostrare al colloquio.

## Fonti

- **Record locale ISEF:** `/tmp/claude-0/-home-user-Radon-and-Bradyseism-model/b712c523-ff6f-58a8-8e52-ea362b608d99/scratchpad/deep/2019-2861.json` — unica fonte usata: anno, nomi, titolo, categoria, paese/stato, premi, tier. Il campo `abstract` di questo file è **scartato** perché appartiene a un altro progetto (vedere avviso iniziale).
- **Nessuna fonte web.** Zero query eseguite: budget WebSearch della sessione esaurito (200/200). Nessun `[web: URL]` compare in questa scheda proprio per questo.
- Le sezioni **Metodo passo per passo**, **Tecniche/strumenti**, **Dati e statistica** e parte di **Perché ha vinto** sono **inferenze dal titolo e conoscenza generale di dominio**, non fatti documentati; le sezioni **Abstract ufficiale** e **Risultati chiave** restano vuote per lo stesso motivo.

### Cosa fare per completare questa scheda
In una sessione con budget di ricerca disponibile: (1) recuperare l'abstract autentico cercando il titolo esatto fra virgolette, con e senza il nome della categoria; (2) correggere il disallineamento dell'abstract nel dataset sorgente, e **verificare se lo stesso errore affligge altri record** dello stesso anno; (3) cercare il titolo del progetto in ambito Regeneron STS / JSHS, dove spesso compare una descrizione metodologica più ricca; (4) non ricercare la persona.
