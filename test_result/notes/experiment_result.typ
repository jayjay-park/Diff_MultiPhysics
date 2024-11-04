// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = [
  #line(start: (25%,0%), end: (75%,0%))
]

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): block.with(
    fill: luma(230), 
    width: 100%, 
    inset: 8pt, 
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.amount
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == "string" {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == "content" {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subrefnumbering: "1a",
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => numbering(subrefnumbering, n-super, quartosubfloatcounter.get().first() + 1))
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => {
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          }

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != "string" {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    new_title_block +
    old_callout.body.children.at(1))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: white, width: 100%, inset: 8pt, body))
      }
    )
}



#let article(
  title: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: (),
  fontsize: 11pt,
  sectionnumbering: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: "1",
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)

  if title != none {
    align(center)[#block(inset: 2em)[
      #text(weight: "bold", size: 1.5em)[#title]
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth,
      indent: toc_indent
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: none
)
#show: doc => article(
  title: [Data Generation],
  date: [2024-09-30],
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)


= Surrogate Modeling for Which System?
<surrogate-modeling-for-which-system>
+ Simplified Geological Carbon Storage (Francis’ paper)
+ Incompressible Navier Stokes

= Twophase flow for the CO2 saturation
<twophase-flow-for-the-co2-saturation>
- We regenerate Francis’ dataset, and additionally compute Fisher Information Matrix as well.
- For the purpose of validation, we currently form full Fisher Infromation Matrix and then compute eigenvector.
- Our next step will be low rank approximation or trace estimation so that we don’t have to form the full matrix.

== Dataset
<dataset>
Our dataset consists of $2000$ pairs of ${ K , S^t (K) }_(t = 1)^8$.

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Example Permeability Model
]
, 
label: 
<fig-K>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 2, gutter: 2em,
  [
#block[
#figure([
#box(width: 80%,image("../../data/Ks_0.png"))
], caption: figure.caption(
position: bottom, 
[
K0
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-surus>


]
],
  [
#block[
#figure([
#box(width: 80%,image("../../data/Ks_1.png"))
], caption: figure.caption(
position: bottom, 
[
K1
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-hanno>


]
],
)
]
)
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Example Saturation Time Series
]
, 
label: 
<fig-S>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 1, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/Snew_series.png"))
], caption: figure.caption(
position: bottom, 
[
Time Series of Saturation of K0
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-S0>


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/Snew_series1.png"))
], caption: figure.caption(
position: bottom, 
[
Time Series of Saturation of K1
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-S1>


]
],
)
]
)
== Fisher Information Matrix
<fisher-information-matrix>
- To find the optimal number of observations, $M$, we visualize eigenvector and vector jacobian product.
- We observe that as $M$ increases, the clearer we see the boundary of the permeabiltiy, which will be more informative during training and inference. #footnote[#link("https://www.overleaf.com/1149716711hxnvfbyfpzvb#a799ce")[Note on Learning Problem];.]
- Given 1 pair of dataset, ${ K , S^t (K) }_(t = 1)^8$, we get a single FIM.

=== Computing Fisher Information Matrix for each datapoint
<computing-fisher-information-matrix-for-each-datapoint>
We consider a realistic scenario when we only have access to samples, but not distribution. When $N$ is number of samples and $X in bb(R)^(d times d)$, neural network model $F_(n n)$ learns mapping from $X_i arrow.r Y_i$. For each pair of ${X_i , Y_i}_(i = 1)^N$, we generate ${F I M_i}_(i = 1)^N$.

- $N$ : number of data points, ${X_i , Y_i}$
- $M$ : number of observation, $Y$

#quote(block: true)[
$ {X_i}_(i = 1)^N tilde.op p_X (X) , med epsilon.alt tilde.op cal(N) (0 , Sigma) , med Sigma = I $ For a single data pair, we generate multiple observations. $ Y_(i , J) = F (X_i) + epsilon.alt_(i , J) , quad w h e r e {epsilon.alt_(i , J)}_(i , J = 1 , 1)^(N , M) $ As we assumed Gaussian, we define likelihood as following. $ p (Y_(i , J) \| X_i) = e^(- 1 / 2 parallel Y_(i , J) - F (X_i) parallel_2^2) $ $ l o g med p (Y_(i , J) \| X_i) approx 1 / Sigma parallel Y_(i , J) - F (X_i) parallel_2^2 $ A FIM for a single data pair $i$ is: $ F I M_i = bb(E)_(Y_(i , { J }_(i = 1)^m) tilde.op p (Y_(i , J) \| X_i)) [(nabla l o g med p (Y_(i , J) \| X_i)) (nabla l o g med p (Y_(i , J) \| X_i))^T] $
]

=== How does FIM change as number of observation increases?
<how-does-fim-change-as-number-of-observation-increases>
FIM is expectation of covariance of derivative of log likelihood. As we expected, we see clearer definition in diagonal relationship as $M$ increases.

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Change in FIM\[:256, :256\] of single data pair ${ K , S^t (K) }_(t = 1)^8$ as number of observation, $M$ increases
]
, 
label: 
<fig-fim>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 3, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/FIM/FIM0_sub0.png"))
], caption: figure.caption(
position: bottom, 
[
M = 1
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/FIM/FIM0_sub0_multi_10.png"))
], caption: figure.caption(
position: bottom, 
[
M = 10
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/FIM/FIM0_sub0_multi_100.png"))
], caption: figure.caption(
position: bottom, 
[
M = 100
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
=== Making Sense of FIM obtained
<making-sense-of-fim-obtained>
#quote(block: true)[
Still, does our FIM make sense? How can we better understand what FIM is representing?
]

Let’s look at the first row of the FIM and reshape it to \[64, 64\].

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Fist, Second, and Third row in FIM
]
, 
label: 
<fig-fimrow>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 3, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=100/FIM_first_row_multi_100.png"))
], caption: figure.caption(
position: bottom, 
[
FIM\[0,:\]
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=100/FIM_sec_row_multi_100.png"))
], caption: figure.caption(
position: bottom, 
[
FIM\[1,:\]
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=100/FIM_third_row_multi_100.png"))
], caption: figure.caption(
position: bottom, 
[
FIM\[2,:\]
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
- Like we expected from the definition of FIM, we observe each plot is just different linear transformation of $nabla l o g p ({ S^t }_(t = 1)^8 \| K)$
- As we will see from below, each rows in FIM is noisy version of its eigenvector.

=== How does eigenvectors of FIM look like as $M$ increases?
<how-does-eigenvectors-of-fim-look-like-as-m-increases>
==== $M = 1$ (Single Observation)
<m-1-single-observation>
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
First three largest eigenvector of FIM
]
, 
label: 
<fig-eig>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 3, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1/FIM_1_first_eig.png"))
], caption: figure.caption(
position: bottom, 
[
First Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1/FIM_1_sec_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Second Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1/FIM_1_third_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Third Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
- Even when FIM is computed with single observation, we see that the largest eigenvector has the most definition in the shape of permeability. Rest of eigenvector looks more like noise.

==== $M = 10$
<m-10>
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
First three largest eigenvector of FIM
]
, 
label: 
<fig-eig10>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 3, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=10/FIM_10_first_eig.png"))
], caption: figure.caption(
position: bottom, 
[
First Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=10/FIM_10_sec_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Second Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=10/FIM_10_third_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Third Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
==== $M = 100$
<m-100>
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
First three largest eigenvector of FIM
]
, 
label: 
<fig-eig100>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 3, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=100/FIM_first_eig.png"))
], caption: figure.caption(
position: bottom, 
[
First Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=100/FIM_sec_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Second Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=100/FIM_third_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Third Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
==== $M = 1000$
<m-1000>
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
First three largest eigenvector of FIM
]
, 
label: 
<fig-eig1000>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 3, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1000/FIM_1000_first_eig.png"))
], caption: figure.caption(
position: bottom, 
[
First Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1000/FIM_1000_sec_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Second Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1000/FIM_1000_third_eig.png"))
], caption: figure.caption(
position: bottom, 
[
Third Eigenvector
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
- As $M$ increases, we observe flow through the channel clearer.
- We see the boundary of permeability gets clearer.
- In general, it gets less noisy.

=== How does vector Jacobian product look like as $M$ increases?
<how-does-vector-jacobian-product-look-like-as-m-increases>
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Normalized Vector Jacobian Product when vector is the largest eigenvector
]
, 
label: 
<fig-eig1000>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 1, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1/FIM_1_vjp.png"))
], caption: figure.caption(
position: bottom, 
[
vjp ($M = 1$)
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=10/FIM_10_vjp.png"))
], caption: figure.caption(
position: bottom, 
[
vjp ($M = 10$)
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=100/FIM_100_vjp.png"))
], caption: figure.caption(
position: bottom, 
[
vjp ($M = 100$)
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../data/N=1000/FIM_1000_vjp.png"))
], caption: figure.caption(
position: bottom, 
[
vjp ($M = 1000$)
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
- We observe that vector Jacobian product looks more like saturation rather than permeability.
- As $M$ increases, scale in color bar also increases.
- One possible conclusion:
  - vjp tells us the location in the spatial distribution (likelihood space) where there exists the largest variation, thus have the most information on parameter.
  - $J^T v$, when $v$ is the largest eigenvector of FIM, is projecting Jacobian onto direction of maximum sensitivity.

= Incompressible Navier Stokes
<incompressible-navier-stokes>
== Dataset
<dataset-1>
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
The first and the last vorticity in a single time series
]
, 
label: 
<fig-vort>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 2, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/input.png"))
], caption: figure.caption(
position: bottom, 
[
Vorticity at $t = 0$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/output.png"))
], caption: figure.caption(
position: bottom, 
[
Vorticity at $t = 40$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
Our dataset consists of 50 pairs of ${ phi^(t - 1) (x_0) , phi^t (x_0) }_(t = 1)^T$, where $T = 44$. Initial vorticities are a Gaussian Random Fields.

== Fisher Information Matrix
<fisher-information-matrix-1>
=== How do we compute FIM?
<how-do-we-compute-fim>
$F I M = (nabla l o g p (phi^t (x_0) \| phi^0 (x_0))) (nabla l o g p (phi^t (x_0) \| phi^0 (x_0)))^T$

- Just means that we are computing FIM with respect to the initial vorticity, $phi^t (x_0)$.

=== How does FIM looks like as $M$ changes?
<how-does-fim-looks-like-as-m-changes>
#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
FIM\[:100, :100\] of varying $M$
]
, 
label: 
<fig-fim_NS>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 1, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_0_9_t=0.png"))
], caption: figure.caption(
position: bottom, 
[
$M = 10$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/100/fim_sub_0_9_t=0.png"))
], caption: figure.caption(
position: bottom, 
[
$M = 100$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
=== Making Sense of FIM obtained
<making-sense-of-fim-obtained-1>
#quote(block: true)[
Still, does our FIM make sense? How can we better understand what FIM is representing?
]

Let’s look at the first row of the Fisher Information Matrix and reshape it to \[64,64\].

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
Comparison of the input parameter with the first element of FIM
]
, 
label: 
<fig-eig100>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 2, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/FIM/past/fim_sub_reshape_0.png"))
], caption: figure.caption(
position: bottom, 
[
FIM\[0, :\]
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/input.png"))
], caption: figure.caption(
position: bottom, 
[
Input Vorticity
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
Also, let’s look at how the first row of the FIM changes as time evolves. When $M = 10$,

#quarto_super(
kind: 
"quarto-float-fig"
, 
caption: 
[
The evolution of the first row of FIM
]
, 
label: 
<fig-fim_NS>
, 
position: 
bottom
, 
supplement: 
"Figure"
, 
subrefnumbering: 
"1a"
, 
subcapnumbering: 
"(a)"
, 
[
#grid(columns: 1, gutter: 2em,
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=0.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 1$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=4.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 5$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=9.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 10$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=14.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 15$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=19.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 20$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=24.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 25$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=29.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 30$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=34.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 35$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=39.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 40$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
  [
#block[
#figure([
#box(width: 100%,image("../../plot/NS_plot/10/fim_sub_reshape_0_9_t=43.png"))
], caption: figure.caption(
position: bottom, 
[
$t = 44$
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)


]
],
)
]
)
== Future Step
<future-step>
+ TODO: Debug NS eigenvector and vjp.
+ TODO: Want to generate the full dataset for Francis’ dataset (which might take 1 or 2 days).
+ TODO: Try it on Jason’s dataset (Now that we fixed the problem with FIM computation, we are optimistic about the experiment, so we want to try it again.)

== Question
<question>
+ What would be the optimal number for observations, $M$ when computing Fisher Information Matrix?

#bibliography("skeleton.bib")

