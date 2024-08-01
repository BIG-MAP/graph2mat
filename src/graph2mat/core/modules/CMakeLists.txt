#add_custom_command(
#  OUTPUT _labels_resort.c
#  DEPENDS _labels_resort.py
#  VERBATIM
#  COMMAND "${CYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/_labels_resort.py" --output-file
#          "${CMAKE_CURRENT_BINARY_DIR}/_labels_resort.c")
#
#python_add_library(_labels_resort MODULE "${CMAKE_CURRENT_BINARY_DIR}/_labels_resort.c"
#                   WITH_SOABI)

#install(TARGETS _labels_resort DESTINATION ${SKBUILD_PROJECT_NAME}/core/modules)
