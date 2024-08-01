add_custom_command(
  OUTPUT _sparse.c
  DEPENDS _sparse.py
  VERBATIM
  COMMAND "${CYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/_sparse.py" --output-file
          "${CMAKE_CURRENT_BINARY_DIR}/_sparse.c")

python_add_library(_sparse MODULE "${CMAKE_CURRENT_BINARY_DIR}/_sparse.c"
                   WITH_SOABI)

install(TARGETS _sparse DESTINATION ${SKBUILD_PROJECT_NAME}/core/data)
