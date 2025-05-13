from datetime import datetime
from pathlib import Path
import pandas as pd
import json

class FileManager:
    @staticmethod
    def carregar_se_existir(caminho_json):
        """Tenta carregar o JSON se existir. Retorna None se o arquivo não for encontrado."""
        path = Path(caminho_json)
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
        return None

    @staticmethod
    def salvar_json(caminho_json, dados):
        """Salva os dados fornecidos em um arquivo JSON."""
        path = Path(caminho_json)
        with path.open("w") as f:
            json.dump(dados, f, indent=4)

    @staticmethod
    def carregar_parametros_do_json(name, caminho_json="melhores_parametros.json"):
        """
        Lê o arquivo JSON e retorna os parâmetros do modelo baseado no 'name'.
        
        :param name: O nome do modelo para buscar no JSON.
        :return: Dicionário dos parâmetros ou dicionário vazio se não encontrado.
        """
        if not name:
            raise ValueError("O parâmetro 'name' deve ser fornecido para carregar os parâmetros.")

        path = Path(caminho_json)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {caminho_json}")

        with path.open("r") as f:
            todos_parametros = json.load(f)

        return todos_parametros.get(name, {})
    
    @staticmethod
    def salvar_resultados(resultados, caminho_csv=None):
        """
        Salva os resultados dos experimentos
        """
        
        df = pd.DataFrame(resultados)
        if not caminho_csv:
            data_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho_csv = f"resultados_experimentos_{data_str}.csv"
        df.to_csv(caminho_csv, index=False)
        print(f"\nResultados salvos em: {caminho_csv}")
        
        return caminho_csv
