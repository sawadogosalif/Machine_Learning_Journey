
plot_ctree<-function(model.ctree){
ggparty(model.ctree) +
  geom_edge() +
  geom_edge_label(mapping = aes(label = substr(breaks_label, start = 1, stop = 15)))+
  geom_node_label(
    line_list = list(
      aes(label = splitvar),
      aes(label = paste("N =", nodesize))
    ),
    line_gpar = list(
      list(size = 13),
      list(size = 10)
    ),
    ids = "inner"
  ) +
  geom_node_label(aes(label = paste0("Node ", id, ", N = ", nodesize)),
                  ids = "terminal", nudge_y = -0.3, nudge_x = 0.01
  ) +
  geom_node_plot(
    gglist = list(
      geom_bar(aes(x = "", fill = cluster),
               position = position_fill(), color = "black"
      ),
      theme_minimal(),
      scale_fill_manual(values = c("brown1","goldenrod", "forestgreen", "turquoise3","steelblue2","hotpink"), guide = FALSE),
      scale_y_continuous(breaks = c(0, 1)),
      xlab(""), ylab("proportion declined"),
      geom_text(aes(
        x = "", group = cluster,
        label = stat(count)
      ),
      stat = "count", position = position_fill(), vjust = 1.7
      )
    ),
    shared_axis_labels = TRUE
  )
}