"""
Análise EDA Ultra-Rápida - Versão Otimizada
Carregamento instantâneo, análise em segundos
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any
import json

warnings.filterwarnings('ignore')

class FastEDA:
    """Análise EDA ultra-rápida sem dependências pesadas"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_fast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise rápida e essencial"""
        
        # Informações básicas (instantâneo)
        basic_info = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_names': list(df.columns)
        }
        
        # Tipos de dados (instantâneo)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        data_types = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols)
        }
        
        # Valores ausentes (rápido)
        missing = df.isnull().sum()
        missing_info = {
            'total_missing': missing.sum(),
            'columns_with_missing': missing[missing > 0].index.tolist(),
            'missing_percentages': {col: (missing[col] / len(df)) * 100 for col in missing.index if missing[col] > 0}
        }
        
        # Estatísticas básicas apenas para numéricas (rápido)
        stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe()
            stats = {
                'summary': desc.to_dict(),
                'correlations': self._fast_correlations(df[numeric_cols]) if len(numeric_cols) > 1 else {}
            }
        
        # Insights rápidos
        insights = self._generate_fast_insights(basic_info, data_types, missing_info, stats)
        
        return {
            'basic_info': basic_info,
            'data_types': data_types,
            'missing_values': missing_info,
            'statistics': stats,
            'insights': insights,
            'sample_data': df.head(3).to_dict('records')
        }
    
    def _fast_correlations(self, df_numeric: pd.DataFrame) -> Dict[str, Any]:
        """Correlações rápidas apenas para variáveis principais"""
        if len(df_numeric.columns) < 2:
            return {}
        
        # Limitar a 10 colunas para velocidade
        cols_to_analyze = df_numeric.columns[:10]
        corr_matrix = df_numeric[cols_to_analyze].corr()
        
        # Encontrar correlações fortes (>0.5)
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        return {
            'strong_correlations': sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True)[:5],
            'matrix_size': f"{len(cols_to_analyze)}x{len(cols_to_analyze)}"
        }
    
    def _generate_fast_insights(self, basic_info, data_types, missing_info, stats) -> List[str]:
        """Gera insights rápidos e essenciais"""
        insights = []
        
        # Insight sobre tamanho
        insights.append(f"📊 Dataset com {basic_info['rows']:,} registros e {basic_info['columns']} variáveis")
        
        # Insight sobre tipos
        insights.append(f"🔢 {data_types['numeric_count']} variáveis numéricas, {data_types['categorical_count']} categóricas")
        
        # Insight sobre missing
        if missing_info['total_missing'] == 0:
            insights.append("✅ Nenhum valor ausente detectado")
        else:
            insights.append(f"⚠️ {missing_info['total_missing']:,} valores ausentes em {len(missing_info['columns_with_missing'])} colunas")
        
        # Insight sobre correlações
        if stats and 'correlations' in stats and stats['correlations']:
            strong_corrs = stats['correlations'].get('strong_correlations', [])
            if strong_corrs:
                insights.append(f"🔗 {len(strong_corrs)} correlações fortes detectadas")
            else:
                insights.append("📈 Nenhuma correlação forte entre variáveis")
        
        # Insight sobre memória
        memory_mb = basic_info['memory_mb']
        if memory_mb < 1:
            insights.append(f"💾 Dataset leve: {memory_mb:.1f} MB")
        elif memory_mb < 100:
            insights.append(f"💾 Dataset médio: {memory_mb:.1f} MB")
        else:
            insights.append(f"💾 Dataset grande: {memory_mb:.1f} MB")
        
        return insights

class FastAgent:
    """Agente ultra-rápido para análise básica"""
    
    def __init__(self):
        self.eda = FastEDA()
        self.df = None
        self.analysis = None
        
    def load_csv_fast(self, file_path: str) -> Dict[str, Any]:
        """Carregamento e análise ultra-rápidos"""
        try:
            # Carregar dados
            self.df = pd.read_csv(file_path)
            
            # Análise rápida
            self.analysis = self.eda.analyze_fast(self.df)
            
            return {
                'success': True,
                'message': f'Carregado: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas',
                'analysis': self.analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Erro: {str(e)}',
                'analysis': None
            }
    
    def answer_fast(self, question: str) -> str:
        """Respostas rápidas para perguntas básicas"""
        if not self.analysis:
            return "❌ Carregue um arquivo CSV primeiro"
        
        q = question.lower()
        
        # Perguntas sobre tamanho
        if any(word in q for word in ['quantas linhas', 'quantos registros', 'tamanho']):
            return f"📊 O dataset tem {self.analysis['basic_info']['rows']:,} linhas e {self.analysis['basic_info']['columns']} colunas"
        
        # Perguntas sobre colunas
        if any(word in q for word in ['quais colunas', 'nomes das colunas', 'variáveis']):
            cols = ', '.join(self.analysis['basic_info']['column_names'])
            return f"📋 Colunas: {cols}"
        
        # Perguntas sobre tipos
        if any(word in q for word in ['tipos', 'tipo das variáveis']):
            dt = self.analysis['data_types']
            return f"🔢 {dt['numeric_count']} numéricas: {', '.join(dt['numeric'][:5])}\n📝 {dt['categorical_count']} categóricas: {', '.join(dt['categorical'][:5])}"
        
        # Perguntas sobre missing
        if any(word in q for word in ['valores ausentes', 'missing', 'nulos']):
            missing = self.analysis['missing_values']
            if missing['total_missing'] == 0:
                return "✅ Nenhum valor ausente"
            else:
                return f"⚠️ {missing['total_missing']:,} valores ausentes em: {', '.join(missing['columns_with_missing'][:3])}"
        
        # Perguntas sobre estatísticas
        if any(word in q for word in ['estatísticas', 'resumo', 'summary']):
            if not self.analysis['statistics']:
                return "📊 Nenhuma variável numérica para estatísticas"
            
            stats = self.analysis['statistics']['summary']
            result = "📊 **Estatísticas Principais:**\n\n"
            
            for col in list(stats.keys())[:3]:  # Apenas 3 primeiras
                col_stats = stats[col]
                result += f"**{col}:**\n"
                result += f"• Média: {col_stats['mean']:.2f}\n"
                result += f"• Min/Max: {col_stats['min']:.2f} / {col_stats['max']:.2f}\n\n"
            
            return result
        
        # Perguntas sobre correlações
        if any(word in q for word in ['correlação', 'correlações', 'relação']):
            corrs = self.analysis['statistics'].get('correlations', {}).get('strong_correlations', [])
            if not corrs:
                return "📈 Nenhuma correlação forte detectada"
            
            result = "🔗 **Correlações Fortes:**\n\n"
            for corr in corrs[:3]:
                result += f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']}\n"
            return result
        
        # Perguntas sobre insights
        if any(word in q for word in ['insights', 'descobertas', 'principais']):
            insights = self.analysis['insights']
            return "💡 **Principais Insights:**\n\n" + "\n".join(insights)
        
        return "🤖 Pergunta não reconhecida. Tente: 'quantas linhas', 'quais colunas', 'estatísticas', 'valores ausentes', 'correlações', 'insights'"

def create_fast_agent():
    """Cria agente ultra-rápido"""
    return FastAgent()
