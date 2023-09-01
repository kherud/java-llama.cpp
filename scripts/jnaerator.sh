#!/bin/bash
cd ..
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
# fix invalid java doc strings (important for javadoc maven plugin)
find src/main/java/de/kherud/llama/foreign \
	-type f \
	-name "*.java" \
	-exec perl -i -pe 's|\* \@see (?!LlamaLibrary\.)|* \@see LlamaLibrary.|g' {} +
cd scripts