#sources of the HAL drivers
HAL_UDMA_SOURCES = \
	$(HAL_TOP)/src/udma/al_hal_udma_main.c \
	$(HAL_TOP)/src/udma/al_hal_udma_config.c \
	$(HAL_TOP)/src/udma/al_hal_udma_iofic.c \
	$(HAL_TOP)/src/udma/al_hal_udma_debug.c \

# TODO need this? 	$(HAL_TOP)/src/udma/al_hal_msg_ipc.c	\

HAL_TPB_SOURCES = \
    $(HAL_TOP)/src/tpb/aws_hal_tpb_pe.c \
    $(HAL_TOP)/src/tpb/aws_hal_tpb_pool.c \
    $(HAL_TOP)/src/tpb/aws_hal_tpb.c \

HAL_SOURCES = $(HAL_UDMA_SOURCES) $(HAL_TPB_SOURCES)

#sources of the init files compiled in the HAL itself
#HAL_INIT_SOURCES_GENERIC = \
#	$(HAL_TOP)/services/err_events/al_err_events_udma.c \

#include path that a HAL user needs
HAL_INCLUDE_PATH = \
	-I$(HAL_TOP)/include/common \
	-I$(HAL_TOP)/include/udma \
	-I$(HAL_TOP)/include/iofic \
    -I$(HAL_TOP)/include/plat_api/sample

#	-I$(HAL_TOP)/include/udma_fast \

