\# PyTorch Deep Learning 

Навчальний проєкт з глибокого навчання на PyTorch, створений на основі практичних матеріалів курсу





\## Реалізовані розділи



У проєкті реалізовано практичні блоки



\* PyTorch Workflow

\* Computer Vision

\* Custom Datasets

\* Going Modular

\* Transfer Learning



\## Dataset



Для класифікації зображень використовується навчальний датасет



\* pizza

\* steak

\* sushi



Датасет автоматично завантажується скриптом



```bash

python -m pytorch\_framework.data.download

```



\## Custom Dataset



У проєкті реалізовано власний клас:



```text

CustomImageDataset

```



Клас успадковується від `torch.utils.data.Dataset` та реалізує методи:



```text

\_\_init\_\_()

\_\_len\_\_()

\_\_getitem\_\_()

```



Для перевірки роботи кастомного датасету:



```bash

python scripts/04\_custom\_datasets.py

```



\## PyTorch Workflow



Базовий workflow тренування моделі лінійної регресії



```bash

python scripts/01\_workflow.py

```



\## Computer Vision



Для демонстрації Computer Vision використовується FashionMNIST і CNN



Запуск:



```bash

python scripts/03\_computer\_vision.py

```



\## TinyVGG



Основна CNN-модель реалізована класом



```text

TinyVGG

```



Модель успадковується від



```text

torch.nn.Module

```



\## Transfer Learning



Для Transfer Learning використовується предтренована EfficientNet-B0



Feature extractor моделі заморожується, а класифікатор замінюється відповідно до кількості класів датасету



\## Evaluation



Для оцінювання збереженої моделі використовується окремий скрипт evaluate.py



Скрипт автоматично визначає архітектуру моделі на основі інформації, збереженої в checkpoint



\## Results



Результати експериментів зберігаються в директорії



```text

runs/

```



Для кожного експерименту створюються



```text

metrics.csv

metrics.json

loss.png

accuracy.png

```





\## Checkpoints



Навчені моделі зберігаються в



```text

checkpoints/

```



Checkpoint містить



\* ваги моделі;

\* стан optimizer;

\* назву моделі;

\* конфігурацію моделі;

\* назви класів.



