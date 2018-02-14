#sources of the HAL drivers
HAL_UDMA_SOURCES = \
	$(HAL_TOP)/src/udma/al_hal_udma_main.c \
	$(HAL_TOP)/src/udma/al_hal_udma_config.c \
	$(HAL_TOP)/src/udma/al_hal_udma_iofic.c \
	$(HAL_TOP)/src/udma/al_hal_udma_debug.c \
	$(HAL_TOP)/src/udma/al_hal_udma_m2m.c \

# TODO need this? 	$(HAL_TOP)/src/udma/al_hal_msg_ipc.c	\

HAL_IOFIC_SOURCES = \
    $(HAL_TOP)/src/iofic/al_hal_iofic.c \

HAL_TPB_SOURCES = \
    $(HAL_TOP)/src/tpb/aws_hal_tpb_pe.c \
    $(HAL_TOP)/src/tpb/aws_hal_tpb_pool.c \
    $(HAL_TOP)/src/tpb/aws_hal_tpb.c \

#HAL_PLAT_SOURCES = \
#    $(HAL_TOP)/src/plat/dummy/dummy_plat_services.c \

HAL_SOURCES = $(HAL_UDMA_SOURCES) $(HAL_TPB_SOURCES) $(HAL_PLAT_SOURCES) $(HAL_IOFIC_SOURCES)

#sources of the init files compiled in the HAL itself
#HAL_INIT_SOURCES_GENERIC = \
#	$(HAL_TOP)/services/err_events/al_err_events_udma.c \

#include path that a HAL user needs

# platform IOMAP, should be generated from something ???
HAL_PLATFORM_INCLUDE = \
    -I$(HAL_TOP)/platform

HAL_MAIN_INCLUDE = \
	-I$(HAL_TOP)/include/common \
	-I$(HAL_TOP)/include/udma \
	-I$(HAL_TOP)/include/iofic \

HAL_PLAT_INCLUDE = \
    -I$(HAL_TOP)/include/plat/dummy \

HAL_INCLUDE_PATH = $(HAL_MAIN_INCLUDE) $(HAL_PLAT_INCLUDE) $(HAL_PLATFORM_INCLUDE)
#	-I$(HAL_TOP)/include/udma_fast \

