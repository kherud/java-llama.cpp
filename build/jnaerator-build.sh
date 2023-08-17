#!/bin/bash
java -jar jnaerator.jar \
	-beanStructs \
	-direct \
	-emptyStructsAsForwardDecls \
	-forceStringSignatures \
	-library llama \
	-package de.kherud.llama.foreign \
	-limitComments \
	-noComments \
	-o ../src/main/java \
	 -preferJavac \
	 -mode Directory \
	 -runtime JNA \
	 llama.h
