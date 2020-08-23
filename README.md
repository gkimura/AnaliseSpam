# Desafio Análise de Spam

Análise exploratória e classificação de mensagens 

### Análises e Implementações

* __src/Análise_de_Spams.ipynb__: Análise exploratória da base de dados senior.csv armazenados em __data/smssenior.csv__.

* __src/app.py__: visualização interativa das análises exploratórias e desempenho apresentado pelos modelos classificadores avaliados.

* __src/classifier.py__: classe contendo o tratamento de balanceamento das classes, divisão do dataset e função de avaliação dos modelos classificadores.

### Execução

* Para a criação do ambiente de execução, segue o comando conda:
	* $ conda env create -f seniorenv.yml
	* $ conda activate senior 

* Para executar a interface de visualização, execute os comandos abaixo. Uma página de navegação será aberta. 
	* $ cd src/
	* $ streamlit run app.py

* Para visualizar as análises exploratórias, execute os comandos abaixo.
	* $ cd src/
	* $ jupyter notebook
	* Selecione o arquivo Análise_de_Spams.ipynb
	* A página com as análises será aberta.



