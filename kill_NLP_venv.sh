#!/usr/bin/env bash

VENVNAME=NLP
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME