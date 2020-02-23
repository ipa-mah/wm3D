#
# Try to find tinyOBJ library and include path.
# Once done this will define
#
# tinyOBJ_FOUND
# tinyOBJ_INCLUDE_PATH
# tinyOBJ_LIBRARY
# 

SET(TINYOBJ_SEARCH_PATHS
	$ENV{tinyobjloader_ROOT}
	${DEPENDENCIES_ROOT}
	/usr			# APPLE
	/usr/local		# APPLE
	/opt/local		# APPLE
)


FIND_PATH(TinyOBJ_INCLUDE_PATH
    NAMES
       tinyOBJ/tiny_obj_loader.h
    PATHS
        ${TINYOBJ_SEARCH_PATHS}
    PATH_SUFFIXES
        include
    DOC
        "The directory where tinyOBJ/tiny_obj_loader.h resides"
)

FIND_LIBRARY( TinyOBJ_LIBRARY
    NAMES 
	tinyOBJ/tinyobjloader.lib
      PATHS
        ${TINYOBJ_SEARCH_PATHS}
    PATH_SUFFIXES
        lib
    DOC
        "The directory where tinyobjloader.lib resides"
  )
SET(TinyOBJ_FOUND "NO")
IF (TinyOBJ_INCLUDE_PATH AND TinyOBJ_LIBRARY)
	SET(TinyOBJ_LIBRARIES ${TinyOBJ_LIBRARY})
	SET(TinyOBJ_FOUND "YES")
    message("EXTERNAL LIBRARY 'tinyobjloader' FOUND")
ELSE()
    message("ERROR: EXTERNAL LIBRARY 'tinyobjloader' NOT FOUND")
ENDIF (TinyOBJ_INCLUDE_PATH AND TinyOBJ_LIBRARY)
