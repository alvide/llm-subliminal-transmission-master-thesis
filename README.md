1. **Clonare la repository**:

   - git clone https://github.com/alvide/llm-subliminal-transmission-master-thesis.git

   - cd llm-subliminal-transmission-master-thesis

2. **Configurare le chiavi:**
   - cp .env.example .env
   
A questo punto deve modificare il file .env appena creato, inserendo il Token Hugging Face che le ho condiviso via email

3. **Lanciare l'esperimento in background:**
   - nohup ./run_tiramisu_72b.sh > run_tiramisu_72b.out 2>&1 &

A questo punto può tranquillamente chiudere il terminale. Lo script si occuperà di fare il build dei container Docker, scaricare il modello, addestrare il Teacher, generare i tweet e valutare gli Student in sequenza.

L'esecuzione durerà diverse ore (non so precisamente quanto). Quando il file run_tiramisu_72b.DONE (che apparirà nella cartella) riporterà la scritta "OK", la pipeline avrà finito e avrà generato un archivio compresso del tipo tiramisu_results_[data_ora].tgz.

Le chiedo di caricare quel file .tgz all'interno di questa cartella Google Drive condivisa: 
