package de.kherud.jllama;

public enum FileType {

	LLAMA_FTYPE_ALL_F32,
	LLAMA_FTYPE_MOSTLY_F16, // except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_0, // except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_1, // except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16, // tok_embeddings.weight and output.weight are F16
	LLAMA_FTYPE_MOSTLY_Q4_2, // support has been removed
	LLAMA_FTYPE_MOSTLY_Q4_3, // support has been removed
	LLAMA_FTYPE_MOSTLY_Q8_0, // except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_0, // except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_1, // except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q2_K,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q3_K_S,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q3_K_M,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q3_K_L,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_K_S,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q4_K_M,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_K_S,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q5_K_M,// except 1d tensors
	LLAMA_FTYPE_MOSTLY_Q6_K,// except 1d tensors

}
