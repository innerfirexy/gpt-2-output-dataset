
:: python "run_entropy.py" --data_dir ..\data\gpt2-generated-from-prompt\ --split $split$ --model $model_name$
:: python .\run_entropy.py --data_dir ..\data\gpt2-generated-from-prompt\ --split story_vary --model gpt2


@echo off
@REM python .\run_entropy.py --data_dir ..\data\gpt2-generated-from-prompt\ --split story_vary --model gpt2
python .\run_entropy.py --data_dir ..\data\gpt2-generated-from-prompt\ --split truenews_35 --model gpt2
python .\run_entropy.py --data_dir ..\data\gpt2-generated-from-prompt\ --split wikitext_35 --model gpt2
python .\run_entropy.py --data_dir ..\data\gpt2-generated-from-prompt\ --split story_vary --model gpt2-xl
python .\run_entropy.py --data_dir ..\data\gpt2-generated-from-prompt\ --split truenews_35 --model gpt2-xl
python .\run_entropy.py --data_dir ..\data\gpt2-generated-from-prompt\ --split wikitext_35 --model gpt2-xl
