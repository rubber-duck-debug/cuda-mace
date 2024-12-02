#[[
 This function wraps the input file with the appropriate string to allow us to 
 static-initialize string variables, for example:
     static const char* CUDA_CODE =
#include "generated/code.cu"
        ;
]]
function(make_includeable INPUT_FILE OUTPUT_FILE)
    if(NOT EXISTS ${INPUT_FILE})
        message(FATAL_ERROR "Error: The input file '${INPUT_FILE}' does not exist.")
    endif()
    file(READ ${INPUT_FILE} content)
    # Format the content to be included as a raw string in C++
    set(content "R\"======(\n${content}\n)======\"")
    # Write the formatted content to the output file
    file(WRITE ${OUTPUT_FILE} "${content}")
endfunction()