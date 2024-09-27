library(rayshader)

#Here, I load a map with the raster package.
loadzip = tempfile() 
localtif = raster::raster("data/raster/FlatTerrainNature.geotiff")
unlink(loadzip)

#And convert it to a matrix:
elmat = raster_to_matrix(localtif)

#We use another one of rayshader's built-in textures:
elmat %>%
    sphere_shade(texture = "desert") %>%
    add_water(detect_water(elmat), color = "desert") %>%
    add_shadow(ray_shade(elmat), 0.5) %>%
    add_shadow(ambient_shade(elmat), 0) %>%
    plot_map()


zscale <- 0.5
fov <- 0
theta <- 135
zoom <- 0.75
phi <- 45
windowsize <- c(1000, 1000)

elmat %>%
    sphere_shade(texture = "imhof1") %>%
    add_water(detect_water(elmat), color = "desert") %>%
    add_shadow(ray_shade(elmat, zscale = zscale), 0.5) %>%
    add_shadow(ambient_shade(elmat), 0) %>%
    plot_3d(elmat, zscale = zscale, fov = fov, theta = theta, zoom = zoom, phi = phi, windowsize = windowsize, 
    water = TRUE, waterdepth = 0, wateralpha = 0.5, watercolor = "lightblue",
    waterlinecolor = "white", waterlinealpha = 0.5)