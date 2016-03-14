FILE(REMOVE_RECURSE
  "CMakeFiles/mark-scripts-executable"
  "../../bin/rotate-movie"
  "../../bin/remove-rotation-metadata"
  "../../bin/extract-frames-from-movie"
  "../../bin/create-movie-from-frames"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/mark-scripts-executable.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
