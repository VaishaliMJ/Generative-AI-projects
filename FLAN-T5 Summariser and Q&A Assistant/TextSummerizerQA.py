"""---------------------------------------------------------------------------------------------------------------
               FLAN-T5 Summariser and Q&A Assistant
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Built a CPU-friendly CLI app that performs abstractive 
                 text summarisation and context-aware Q&A using Google/flan-t5-small, 
                 ideal for classroom demos and quick local workflows 
                 (no training required)
-------------------------------------------------------------------------------------------------------------------"""
######################################################################################
# Constants and imports
######################################################################################
import os
import tensorflow as TF

os.environ["TOKENIZERS_PARALLELISM"]="false"


from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
            #AutoTokenizer : A autoktokenizer loader 
            # that automatically picks the right tokenizer for model you choose
            #AutoModelForSeq2SeqLM : A pre-trained model loader for sequence-to-sequence
            #Language models(Seq2SeqLM)
######################################################################################

BORDER="-"*50
# Choose a instruction-tuned model
MODEL_NAME="google/flan-t5-small"
#   A light-weight version of FLAN-T5
#   About 80 million parameters     
print(f"FLAN-T5_Summarizer_Q&A_Assistant{MODEL_NAME} model loading.....")

#Load tokenizer
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

#Load the sequence-to-Sequence model
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
       
######################################################################################

###############################################################################
#   Function        :   AcceptSummaryText
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   This function summarizes input text
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Feb 2026
###############################################################################
def AcceptSummaryText():
    

    lines=[]
    while True:
        inputText=input()
        #Stop when user enters blank line or Enter key
        if not inputText.strip():
            break
        lines.append(inputText)
    #Joins lines into a text block
    textBlock="\n".join(lines).strip()
    
    return textBlock
###############################################################################
#   Function        :   Run_flan
#   Input Params    :   text:str,max_new_tokens:int
#   Output Params   :   -
#   Description     :   This function is used for summarization.
#                       It creates prompt with 4-5 bullet points
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Feb 2026
##############################################################################
def Run_flan(prompt:str,max_new_tokens:int=256)->str:   
    #Tokenize the input prompt;return PyTorch tensors ;Truncate if too long
    inputs=tokenizer(prompt,return_tensors="pt",truncation=True)

    #Generation
    #Generate text from the model with light sampling for naturalness
    output=model.generate(
        **inputs,                       #Pass Tokenized inputs(input_ids,attention_mask)
        max_new_tokens=max_new_tokens,  #How many tokens to generate
        do_sample=True,                 #Enable random sampling
        top_p=0.9,                      #nucleus sampling:only consider tokens in the top 90%probablity mass
        temperature=0.7                 #Control_randomness (lower=safer/more deterministic)

    ) 
    #Decode token ids back into clean string
    #Example Ids    :       [71,867,1234,42,1]
    #Text           :       "Hello,How are you?"
    return tokenizer.decode(output[0],skip_special_tokens=True).strip() 
   
##############################################################################
#   Function        :   Summarize_text
#   Input Params    :   text:str
#   Output Params   :   Summarized text:str
#   Description     :   This function is used for summarization.
#                       It creates prompt with 4-5 bullet points
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Feb 2026
##############################################################################
def Summarize_text(text:str)->str:
    #Prompt template instructing the model to produce 4-6 bullet points
    prompt=f"Summarize the following text in 4-6 bullet points:\n\n{text}"

    #Allow a slightly longer output for bullet list
    return Run_flan(prompt,max_new_tokens=160)        
###############################################################################
#   Function        :   load_context
#   Input Params    :   path:str="context.txt"
#   Output Params   :   text:str
#   Description     :   This function is used to load contents from out local file
#                       and returns the complete file content in one string
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Feb 2026
##############################################################################
def load_context(path:str="context.txt")->str:
    try:
        #Read entire file as a single string
        with open (path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
###############################################################################
#   Function        :   answer_from_context
#   Input Params    :   question:str,context:str
#   Output Params   :   text:str
#   Description     :   This function ask FLAN to answer using only given context
#                       If the answer is not present ,ask it to say "Not found"
#    
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Feb 2026
##############################################################################
def answer_from_context(question:str,context:str)->str:
    if not context.strip():
        return "Context file not found or empty .Create 'Context.txt' first"
    #Create a Strict prompt for FLAN-T5

    prompt=("You are a helpful assistant.Answer the question only using the context\n"
            "If the answer is not in the context,reply exactly:Not found\n\n"
            f"Context:\n{context}\n\n"
            f"Question:{question}\nAnswer:\n")
    
    #Generate concise answer grounded in the provided notes
    return Run_flan(prompt,max_new_tokens=120)    
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point
#   Author          :   Vaishali M Jorwekar
#   Date            :   12 Feb 2026
###############################################################################
def main():
    print(BORDER)
    print("----------- FLAN-T5 Model Text summarizer and QA Model-----------------")
    print(BORDER)
    print("\n1.Summarize the data")
    print("\n2.Question and Answer over local context.txt")
    print("\n0.Exit")
    print(BORDER)
    while True:
        choice=input("\nChoose an option (1/2/0):").strip()
        
        if choice=="0":
            print("Thank you for using FLAN-T5 Model Text summarizer and QA Model")
            break
        
        elif choice == "1":
            #Collect muliple lines of text for summarization
            print("\nPaste text to summarize . End with a blank line...")
            textBlock=AcceptSummaryText()
            
            
            if not textBlock:
                print("FLAN-T5 says: No text received...")
                continue
            #Run summarization and print the result
            print(BORDER)
            print("Summary Generated by FLAN-T5 model:\n")
            print(Summarize_text(textBlock))
        elif choice=="2":
            #Load context from local file context.txt
            ctx=load_context("context.txt")
            if not ctx.strip():
                #Help user if context.txt is missing/empty
                print(f"Missing 'context.txt' .create in the same folder and try again...")
                continue
            # Ask a question related to the provided context 
            q=input("\n Ask a question about your context to  FLAN Model:").strip()
            if not q:
                print("No question received...")
                continue
            #Generate answer grounded  only in the context
            print("\nAnswer from  FLAN Model:")
            print(answer_from_context(q,ctx))
        else:
            ("Please choose 1/2/0")   
######################################################################################
#   STARTER
######################################################################################
if __name__=="__main__":
    main()
    
    