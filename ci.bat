@echo off

echo fomating ..
black src tests notebooks

echo lint ..
flake8 src tests

echo tests .. 
pytest tests