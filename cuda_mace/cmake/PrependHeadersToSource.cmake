
#[[
 This function wraps prepends a list of headers to an input file. Used as a preprocessing
 step to prepare the code.cu file for NVRTC JIT compilation.
]]
function(prepend_headers_to_source INPUT_FILE OUTPUT_FILE HEADERS)
    # Read the current content of the source file
    if(NOT EXISTS ${INPUT_FILE})
        message(FATAL_ERROR "Error: The input file '${INPUT_FILE}' does not exist.")
    endif()
    file(READ ${INPUT_FILE} CONTENTS)
    # Initialize a variable to hold the accumulated headers
    set(HEADER_CONTENTS "")
    # Loop through each header and append its contents to HEADER_CONTENTS
    foreach(HEADER IN LISTS HEADERS)
        if(NOT EXISTS ${INPUT_FILE})
            message(FATAL_ERROR "Error: The input file '${INPUT_FILE}' does not exist.")
        endif()
        file(READ ${HEADER} SINGLE_HEADER_CONTENT)
        set(HEADER_CONTENTS "${HEADER_CONTENTS}\n${SINGLE_HEADER_CONTENT}")
    endforeach()
    # Prepend the accumulated headers to the original source file content
    set(NEW_CONTENTS "${HEADER_CONTENTS}\n${CONTENTS}")
    # Write the new contents back to the source file
    file(WRITE ${OUTPUT_FILE} "${NEW_CONTENTS}")
endfunction()