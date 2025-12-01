## Config Readme

Bu dosya, `interactive_analyzer_with_metrics.py` betiğini **config dosyası** kullanarak nasıl çalıştırabileceğini açıklar.

### 1. Varsayılan config ile çalıştırma

`config.json` ile aynı klasörde (`autogen_clustering`) olduğun sürece:

```bash
python autogen_clustering/interactive_analyzer_with_metrics.py
```

Bu komut:
- `config.json` içindeki `run_config` bölümünü okur
- Oradaki `graph_file`, `dependency_rsf`, `clustering_model`, `clusters`, `iterations`, `output` vb. ayarları kullanır.

### 2. Config + bazı parametreleri override etme

Config’teki değerleri koruyup sadece birkaçını değiştirmek için komut satırında override edebilirsin:

```bash
python autogen_clustering/interactive_analyzer_with_metrics.py ^
  --clusters 15 ^
  --iterations 2
```

Bu durumda:
- Diğer tüm ayarlar `config.json`’dan gelir
- Sadece `clusters = 15` ve `iterations = 2` olarak güncellenir.

PowerShell yerine bash kullanıyorsan, aynı komut:

```bash
python autogen_clustering/interactive_analyzer_with_metrics.py \
  --clusters 15 \
  --iterations 2
```

### 3. Graph dosyasını komut satırından vermek

Config’teki `graph_file` değerini göz ardı edip, doğrudan komut satırından verebilirsin:

```bash
python autogen_clustering/interactive_analyzer_with_metrics.py ^
  ./dataset/bash/graph.pkl
```

Burada:
- Argüman olarak verdiğin `./dataset/bash/graph.pkl`, `run_config.graph_file` değerinin yerini alır.

### 4. Farklı bir config dosyası kullanmak

Örneğin `my_config.json` isimli kendi config dosyanı kullanmak istersen:

```bash
python autogen_clustering/interactive_analyzer_with_metrics.py ^
  --config my_config.json
```

Bu durumda:
- `autogen_clustering/my_config.json` okunur
- Oradaki `run_config` değerleri baz alınır.

### 5. Modelleri ve host’u override etme

Config’teki modelleri ve Ollama host’unu komut satırından geçici olarak değiştirebilirsin:

```bash
python autogen_clustering/interactive_analyzer_with_metrics.py ^
  --clustering-model gpt-os:120b-cloud ^
  --metrics-model gpt-os:120b-cloud ^
  --host http://localhost:11434
```

### 6. Örnek `config.json` şablonu

`config.json` için örnek bir `run_config` bölümü:

```json
{
  "run_config": {
    "graph_file": "./dataset/bash/graph.pkl",
    "dependency_rsf": "./dataset/bash/bash-dependency_compact.rsf",
    "reference_rsf": null,
    "clustering_model": "gpt-os:120b-cloud",
    "metrics_model": "gpt-os:120b-cloud",
    "clusters": 10,
    "iterations": 1,
    "output": "clusters_with_metrics.json",
    "ollama_host": "http://localhost:11434",
    "timeout": 600,
    "temperature": 0
  }
}
```

Komut satırından verdiğin her parametre (`--clusters`, `--host`, `--output` vb.), config’teki ilgili değeri **override** eder.


