#!/bin/bash
jnaerator \
	-beanStructs \
	-direct \
	-emptyStructsAsForwardDecls \
	-forceStringSignatures \
	-library llama \
	-package de.kherud.llama.foreign \
	-limitComments \
	-skipDeprecated \
	-noComments \
	-o src/main/java \
	 -preferJavac \
	 -mode Directory \
	 -runtime JNA \
	 -f \
	 llama.h
# remove erroneous method signatures
find src/main/java/de/kherud/llama/foreign \
 	-type f \
 	-name "*.java" \
 	-exec sed -i 's/protected List<? > getFieldOrder()/@Override\n\tprotected List<String> getFieldOrder()/g' {} +
# remove unknown imports
find src/main/java/de/kherud/llama/foreign \
	-type f \
	-name "*.java" \
	-exec sed -i -E '/import com\.ochafik\.lang\.jnaerator\.runtime\.NativeSize(ByReference)?;/d' {} +
# remove deprecated
find src/main/java/de/kherud/llama/foreign -type f -name "*.java" -exec sed -i '/@Deprecated/{N; d;}' {} +
